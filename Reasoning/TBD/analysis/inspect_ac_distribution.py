"""Debug script to inspect AC probability distribution when rule applies"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
REASONER_DIR = ROOT / "tre_reasoner"
sys.path.insert(0, str(REASONER_DIR))

from data import TRECachedDataset, tre_collate_cached
from data_sampler import DocBatchSampler
from model import create_model
from conf_loader import load_runtime_config
from transformers import AutoModel, BertTokenizerFast

BASE = ROOT / "TBD_old"
CHECKPOINT = BASE / "checkpoint" / "TBD_baseline_reasoning_model.pt"
CONF = ROOT / "script" / "config" / "TBD.conf"
RULES = ROOT / "rules" / "rules_TBD_old.txt"

REL = {"AFTER": 0, "BEFORE": 1, "INCLUDES": 2, "IS_INCLUDED": 3, "SIMULTANEOUS": 4, "VAGUE": 5}
I2R = {v: k for k, v in REL.items()}
NON_VAGUE_IDXS = [idx for name, idx in REL.items() if name != "VAGUE"]

TRANS = {
    ("BEFORE", "BEFORE"): "BEFORE",
    ("AFTER", "AFTER"): "AFTER",
    ("SIMULTANEOUS", "BEFORE"): "BEFORE",
    ("BEFORE", "SIMULTANEOUS"): "BEFORE",
    ("SIMULTANEOUS", "AFTER"): "AFTER",
    ("AFTER", "SIMULTANEOUS"): "AFTER",
    ("SIMULTANEOUS", "SIMULTANEOUS"): "SIMULTANEOUS",
    ("INCLUDES", "INCLUDES"): "INCLUDES",
    ("IS_INCLUDED", "IS_INCLUDED"): "IS_INCLUDED",
}


def build_model(device: torch.device):
    _, model_cfg = load_runtime_config("TBD", str(CONF))
    bert_name = model_cfg.bert_model
    tokenizer = BertTokenizerFast.from_pretrained(bert_name)
    bert = AutoModel.from_pretrained(bert_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
    bert.resize_token_embeddings(len(tokenizer))

    model = create_model(
        mode="baseline_reasoning",
        bert=bert,
        num_classes=6,
        rule_file_path=str(RULES),
        relation_map=REL,
        num_types=1,
        step_size=model_cfg.step_size,
        lambda_kl=model_cfg.lambda_kl,
        smooth_tau=model_cfg.smooth_tau,
        max_steps=model_cfg.max_steps,
        tol=model_cfg.tol,
        use_deq=model_cfg.use_deq,
        output_option=model_cfg.output_option,
        learn_rule_weights=model_cfg.learn_rule_weights,
        initial_rule_weight=model_cfg.initial_rule_weight,
        rule_chunk_size=model_cfg.rule_chunk_size,
    )

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(device).eval()
    return model


def top_non_vague_label(probs):
    local_idx = int(np.argmax(probs[NON_VAGUE_IDXS]))
    return I2R[NON_VAGUE_IDXS[local_idx]]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = build_model(device)

dataset_cfg, _ = load_runtime_config("TBD", str(CONF))
dataset = TRECachedDataset(dataset_cfg.test_token_path, dataset_cfg.test_jsonl_path)
loader = DataLoader(
    dataset,
    batch_sampler=DocBatchSampler(dataset.doc_ids, batch_size=256, shuffle=False),
    collate_fn=tre_collate_cached,
)

for batch_idx, batch in enumerate(loader, start=1):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    word_marks = batch.get("word_marks")
    e1_marks = batch.get("e1_marks")
    e2_marks = batch.get("e2_marks")
    e1_type_ids = batch.get("e1_type_ids")
    e2_type_ids = batch.get("e2_type_ids")

    if isinstance(word_marks, torch.Tensor):
        word_marks = word_marks.to(device)
    if isinstance(e1_marks, torch.Tensor):
        e1_marks = e1_marks.to(device)
    if isinstance(e2_marks, torch.Tensor):
        e2_marks = e2_marks.to(device)
    if isinstance(e1_type_ids, torch.Tensor):
        e1_type_ids = e1_type_ids.to(device)
    if isinstance(e2_type_ids, torch.Tensor):
        e2_type_ids = e2_type_ids.to(device)

    with torch.no_grad():
        P = model.neural(input_ids, attention_mask, word_marks, e1_marks, e2_marks)["P"].to(device)

    e1_ids = batch["e1_ids"]
    e2_ids = batch["e2_ids"]
    doc_ids = batch["doc_ids"]
    labels = batch.get("labels", None)
    
    if isinstance(doc_ids, torch.Tensor):
        doc_ids = doc_ids.tolist()

    pair_to_idx = {}
    successors = {}
    for i, (doc, e1, e2) in enumerate(zip(doc_ids, e1_ids, e2_ids)):
        pair_to_idx.setdefault(doc, {})[(e1, e2)] = i
        successors.setdefault(doc, {}).setdefault(e1, []).append(e2)

    ac_prob_when_rule_applies = []
    
    for doc, pairs in pair_to_idx.items():
        for (a, b), idx_ab in pairs.items():
            for c in successors[doc].get(b, []):
                if a == c or (a, c) not in pairs:
                    continue
                    
                idx_bc = pairs[(b, c)]
                idx_ac = pairs[(a, c)]

                if labels is None:
                    continue
                
                gt_ab = int(labels[idx_ab].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_ab])
                gt_bc = int(labels[idx_bc].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_bc])
                
                rel_ab_gt = I2R[gt_ab]
                rel_bc_gt = I2R[gt_bc]

                # Get predictions at t=0
                p_ab0 = P[idx_ab].detach().cpu().numpy()
                p_bc0 = P[idx_bc].detach().cpu().numpy()
                p_ac0 = P[idx_ac].detach().cpu().numpy()
                
                pred_ab0 = top_non_vague_label(p_ab0)
                pred_bc0 = top_non_vague_label(p_bc0)

                # Skip if LHS not correct
                if pred_ab0 != rel_ab_gt or pred_bc0 != rel_bc_gt:
                    continue

                expected_ac = TRANS.get((rel_ab_gt, rel_bc_gt), None)
                if expected_ac is None:
                    continue
                    
                prob_expected = float(p_ac0[REL[expected_ac]])
                ac_prob_when_rule_applies.append(prob_expected)

    print(f"\nBatch {batch_idx}: {len(ac_prob_when_rule_applies)} rule-applicable triplets found")
    if ac_prob_when_rule_applies:
        probs = np.array(ac_prob_when_rule_applies)
        print(f"  AC expected label probability distribution:")
        print(f"    - Min: {probs.min():.4f}")
        print(f"    - Median: {np.median(probs):.4f}")
        print(f"    - Mean: {probs.mean():.4f}")
        print(f"    - Max: {probs.max():.4f}")
        print(f"  - < 0.3: {(probs < 0.3).sum()}")
        print(f"  - < 0.4: {(probs < 0.4).sum()}")
        print(f"  - < 0.5: {(probs < 0.5).sum()}")
        print(f"  - >= 0.5: {(probs >= 0.5).sum()}")
    break  # Only first batch
