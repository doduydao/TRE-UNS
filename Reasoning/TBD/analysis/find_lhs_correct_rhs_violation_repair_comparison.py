import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
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

REL = {
    "AFTER": 0,
    "BEFORE": 1,
    "INCLUDES": 2,
    "IS_INCLUDED": 3,
    "SIMULTANEOUS": 4,
    "VAGUE": 5,
}
I2R = {v: k for k, v in REL.items()}
NON_VAGUE_IDXS = [idx for name, idx in REL.items() if name != "VAGUE"]

# Transitivity rules: if (A,B)=X and (B,C)=Y then (A,C) SHOULD be Z
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

TARGET = {"BEFORE", "AFTER", "SIMULTANEOUS"}
CAPTURE = [0, 1, 5, 10, 50]


def top_non_vague_label(probs):
    local_idx = int(np.argmax(probs[NON_VAGUE_IDXS]))
    top_idx = NON_VAGUE_IDXS[local_idx]
    return I2R[top_idx], float(probs[top_idx])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["trained", "untrained"], default="trained",
                        help="Use trained or untrained model")
    parser.add_argument("--lhs-confidence", type=float, default=0.7,
                        help="Minimum confidence for BOTH LHS pairs at t=0")
    parser.add_argument("--rhs-initial-low", type=float, default=0.55,
                        help="Maximum probability for RHS expected label at t=0 (violation)")
    parser.add_argument("--rhs-repair-min", type=float, default=0.01,
                        help="Minimum increase in RHS probability through reasoning")
    return parser.parse_args()


def build_model(device: torch.device, use_trained: bool = True):
    """Build model with option to use trained weights or random initialization"""
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

    if use_trained:
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    
    model.to(device).eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model Type: {args.model_type}")

    use_trained = (args.model_type == "trained")
    model = build_model(device, use_trained=use_trained)
    reasoning = model.reasoning

    dataset_cfg, _ = load_runtime_config("TBD", str(CONF))
    dataset = TRECachedDataset(dataset_cfg.test_token_path, dataset_cfg.test_jsonl_path)
    loader = DataLoader(
        dataset,
        batch_sampler=DocBatchSampler(dataset.doc_ids, batch_size=256, shuffle=False),
        collate_fn=tre_collate_cached,
    )

    rows = []
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

        ctx, pre = reasoning._prepare_context(P, list(zip(e1_ids, e2_ids)), doc_ids, e1_type_ids, e2_type_ids)
        Q = P.clone().detach()
        snapshots = {0: P.detach().cpu().clone()}

        clip_val = getattr(reasoning, "grad_clip_value", 10.0)
        for step in range(1, max(CAPTURE) + 1):
            with torch.enable_grad():
                Q_req = Q.detach().requires_grad_(True)
                out = reasoning._compute_reason_energy(P, Q_req, ctx, pre)
                (gq,) = torch.autograd.grad(out["E_re"], Q_req, create_graph=False)
                gq = torch.clamp(gq, -clip_val, clip_val)
                Q = torch.softmax(torch.log(Q + 1e-12) - reasoning.step_size * gq, dim=-1).detach()
            if step in CAPTURE:
                snapshots[step] = Q.cpu().clone()

        # Count triplets and filter by criteria
        triplet_count = 0
        crit_ab_correct = 0
        crit_bc_correct = 0
        crit_has_rule = 0
        crit_ac_low_prob = 0
        crit_ac_repaired = 0

        for doc, pairs in pair_to_idx.items():
            for (a, b), idx_ab in pairs.items():
                for c in successors[doc].get(b, []):
                    if a == c or (a, c) not in pairs:
                        continue
                    triplet_count += 1
                    idx_bc = pairs[(b, c)]
                    idx_ac = pairs[(a, c)]

                    if labels is None:
                        continue
                    
                    gt_ab = int(labels[idx_ab].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_ab])
                    gt_bc = int(labels[idx_bc].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_bc])
                    
                    rel_ab_gt = I2R[gt_ab]
                    rel_bc_gt = I2R[gt_bc]

                    # Get predictions at t=0
                    p_ab0 = snapshots[0][idx_ab].numpy()
                    p_bc0 = snapshots[0][idx_bc].numpy()
                    p_ac0 = snapshots[0][idx_ac].numpy()
                    
                    pred_ab0, _ = top_non_vague_label(p_ab0)
                    pred_bc0, _ = top_non_vague_label(p_bc0)

                    # Criterion 1: LHS (A,B) prediction is correct
                    if pred_ab0 != rel_ab_gt:
                        continue
                    crit_ab_correct += 1

                    # Criterion 2: LHS (A,B) has high confidence
                    conf_ab0 = float(p_ab0[REL[rel_ab_gt]])
                    if conf_ab0 < args.lhs_confidence:
                        continue

                    # Criterion 3: LHS (B,C) prediction is also correct
                    if pred_bc0 != rel_bc_gt:
                        continue
                    crit_bc_correct += 1

                    # Criterion 4: LHS (B,C) has high confidence
                    conf_bc0 = float(p_bc0[REL[rel_bc_gt]])
                    if conf_bc0 < args.lhs_confidence:
                        continue

                    # Check if rule applies: (A,B) ∘ (B,C) → (A,C) expected
                    expected_ac = TRANS.get((rel_ab_gt, rel_bc_gt), None)
                    if expected_ac is None or expected_ac not in TARGET:
                        continue
                    crit_has_rule += 1

                    # Criterion 5: RHS (A,C) has LOW confidence (violation)
                    prob_ac_expected_t0 = float(p_ac0[REL[expected_ac]])
                    if prob_ac_expected_t0 > args.rhs_initial_low:
                        continue
                    crit_ac_low_prob += 1

                    # Get RHS (A,C) probability at final timestep
                    p_ac_final = snapshots[CAPTURE[-1]][idx_ac].numpy()
                    prob_ac_expected_final = float(p_ac_final[REL[expected_ac]])

                    # Criterion 6: RHS probability INCREASED through reasoning
                    rhs_increase = prob_ac_expected_final - prob_ac_expected_t0
                    if rhs_increase < args.rhs_repair_min:
                        continue
                    crit_ac_repaired += 1

                    pred_ac0, _ = top_non_vague_label(p_ac0)
                    pred_ac_final, _ = top_non_vague_label(p_ac_final)

                    rows.append({
                        "doc": doc,
                        "a": a,
                        "b": b,
                        "c": c,
                        "rule": f"{rel_ab_gt}∘{rel_bc_gt}→{expected_ac}",
                        "lhs_ab_conf": conf_ab0,
                        "lhs_bc_conf": conf_bc0,
                        "rhs_prob_t0": prob_ac_expected_t0,
                        "rhs_prob_final": prob_ac_expected_final,
                        "rhs_increase": rhs_increase,
                        "pred_ac0": pred_ac0,
                        "pred_ac_final": pred_ac_final,
                    })

        print(f"Processed batch {batch_idx}, checked {triplet_count} triplets")
        print(f"  Criteria breakdown:")
        print(f"    - AB correct: {crit_ab_correct}")
        print(f"    - BC correct: {crit_bc_correct}")
        print(f"    - Has valid rule: {crit_has_rule}")
        print(f"    - AC low prob (violation): {crit_ac_low_prob}")
        print(f"    - AC successfully repaired: {crit_ac_repaired}")

    if not rows:
        print("\n✗ No candidates found!")
        return

    print(f"\n✓ Found {len(rows)} candidates!")
    print(f"Thresholds: lhs_conf={args.lhs_confidence}, rhs_initial_low={args.rhs_initial_low}, rhs_repair_min={args.rhs_repair_min}\n")

    df = pd.DataFrame(rows)
    df = df.sort_values("rhs_increase", ascending=False)

    print("Top 10 (sorted by RHS repair amount):")
    print(df.head(10).to_string(index=False))

    output_file = f"lhs_correct_rhs_violation_repair_{args.model_type}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {Path(output_file).resolve()}")


if __name__ == "__main__":
    main()
