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
CAPTURE = [0, 10, 50, 100, 500, 1000]


def top_non_vague_label(probs):
    local_idx = int(np.argmax(probs[NON_VAGUE_IDXS]))
    top_idx = NON_VAGUE_IDXS[local_idx]
    return I2R[top_idx], float(probs[top_idx])


def top_full_label(probs):
    idx = int(np.argmax(probs))
    return I2R[idx], float(probs[idx])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs-before-min", type=float, default=0.45,
                        help="Minimum BEFORE probability on both LHS pairs at t=0")
    parser.add_argument("--ac-before-max-t0", type=float, default=0.50,
                        help="Maximum BEFORE probability on AC at t=0")
    parser.add_argument("--ac-before-min-inc", type=float, default=0.03,
                        help="Minimum BEFORE probability increase from t=0 to corrected step")
    parser.add_argument("--prediction-mode", choices=["non-vague", "full"], default="non-vague",
                        help="Use argmax over non-vague classes only, or all 6 classes")
    return parser.parse_args()


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


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(device)
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
        labels = batch.get("labels")

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

        for doc, pairs in pair_to_idx.items():
            for (a, b), idx_ab in pairs.items():
                for c in successors[doc].get(b, []):
                    if a == c or (a, c) not in pairs:
                        continue
                    if labels is None:
                        continue

                    idx_bc = pairs[(b, c)]
                    idx_ac = pairs[(a, c)]

                    gt_ab = int(labels[idx_ab].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_ab])
                    gt_bc = int(labels[idx_bc].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_bc])
                    gt_ac = int(labels[idx_ac].item()) if isinstance(labels, torch.Tensor) else int(labels[idx_ac])

                    rel_ab_gt = I2R[gt_ab]
                    rel_bc_gt = I2R[gt_bc]
                    rel_ac_gt = I2R[gt_ac]

                    # Focus only on BEFORE-BEFORE-BEFORE as requested.
                    if not (rel_ab_gt == "BEFORE" and rel_bc_gt == "BEFORE" and rel_ac_gt == "BEFORE"):
                        continue

                    p_ab0 = snapshots[0][idx_ab].numpy()
                    p_bc0 = snapshots[0][idx_bc].numpy()
                    p_ac0 = snapshots[0][idx_ac].numpy()

                    lhs_ab_before = float(p_ab0[REL["BEFORE"]])
                    lhs_bc_before = float(p_bc0[REL["BEFORE"]])
                    ac_before_t0 = float(p_ac0[REL["BEFORE"]])

                    if lhs_ab_before < args.lhs_before_min or lhs_bc_before < args.lhs_before_min:
                        continue
                    if ac_before_t0 > args.ac_before_max_t0:
                        continue

                    top_label = top_non_vague_label if args.prediction_mode == "non-vague" else top_full_label

                    pred_ac0, _ = top_label(p_ac0)
                    if pred_ac0 == "BEFORE":
                        continue

                    corrected_step = None
                    pred_ac_t = None
                    ac_before_t = None
                    for step in CAPTURE[1:]:
                        p_act = snapshots[step][idx_ac].numpy()
                        pred_step, _ = top_label(p_act)
                        if pred_step == "BEFORE":
                            corrected_step = step
                            pred_ac_t = pred_step
                            ac_before_t = float(p_act[REL["BEFORE"]])
                            break

                    if corrected_step is None:
                        continue

                    inc = ac_before_t - ac_before_t0
                    if inc < args.ac_before_min_inc:
                        continue

                    rows.append({
                        "doc": doc,
                        "a": a,
                        "b": b,
                        "c": c,
                        "lhs_ab_before_t0": lhs_ab_before,
                        "lhs_bc_before_t0": lhs_bc_before,
                        "ac_before_t0": ac_before_t0,
                        "pred_ac_t0": pred_ac0,
                        "pred_ac_t": pred_ac_t,
                        "corrected_step": corrected_step,
                        "ac_before_t": ac_before_t,
                        "ac_before_inc": inc,
                    })

        print(f"Processed batch {batch_idx}")

    out = pd.DataFrame(rows)
    if out.empty:
        print("No candidate found with current thresholds.")
        return

    out = out.sort_values(
        ["ac_before_inc", "lhs_ab_before_t0", "lhs_bc_before_t0", "corrected_step"],
        ascending=[False, False, False, True],
    )

    out_path = BASE / "analysis" / "before_wrong_to_correct_anyT.csv"
    out.to_csv(out_path, index=False)

    print(f"Candidates: {len(out)}")
    print(f"Prediction mode: {args.prediction_mode}")
    print("Top 10:")
    print(out.head(10).to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
