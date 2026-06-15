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
    return I2R[NON_VAGUE_IDXS[local_idx]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gap0", type=float, default=0.20,
                        help="Minimum initial violation gap: LHS_t0 - RHS_t0")
    parser.add_argument("--min-rhs-inc", type=float, default=0.05,
                        help="Minimum RHS probability increase from t0 to t50")
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
                    idx_bc = pairs[(b, c)]
                    idx_ac = pairs[(a, c)]

                    p_ab0 = snapshots[0][idx_ab].numpy()
                    p_bc0 = snapshots[0][idx_bc].numpy()
                    rel_ab0 = top_non_vague_label(p_ab0)
                    rel_bc0 = top_non_vague_label(p_bc0)
                    expected = TRANS.get((rel_ab0, rel_bc0))
                    if expected is None or expected not in TARGET:
                        continue

                    rid = REL[expected]
                    rhs = [float(snapshots[s][idx_ac][rid]) for s in CAPTURE]
                    lhs = [
                        min(
                            float(snapshots[s][idx_ab][REL[rel_ab0]]),
                            float(snapshots[s][idx_bc][REL[rel_bc0]]),
                        )
                        for s in CAPTURE
                    ]
                    gap = [l - r for l, r in zip(lhs, rhs)]

                    rhs_inc = rhs[-1] - rhs[0]
                    gap_dec = gap[0] - gap[-1]
                    mono_hits = sum(rhs[i] >= rhs[i - 1] for i in range(1, len(rhs)))

                    # Enforce user's criterion: must start with low RHS vs LHS, then RHS must increase.
                    if gap[0] < args.min_gap0:
                        continue
                    if rhs_inc < args.min_rhs_inc:
                        continue

                    rows.append(
                        {
                            "doc": doc,
                            "a": a,
                            "b": b,
                            "c": c,
                            "premise_step0": f"{rel_ab0} o {rel_bc0}",
                            "expected": expected,
                            "lhs_t0": lhs[0],
                            "rhs_t0": rhs[0],
                            "gap_t0": gap[0],
                            "lhs_t50": lhs[-1],
                            "rhs_t50": rhs[-1],
                            "gap_t50": gap[-1],
                            "rhs_increase": rhs_inc,
                            "gap_decrease": gap_dec,
                            "monotonic_hits": mono_hits,
                            "rhs_path": "|".join(f"{x:.3f}" for x in rhs),
                            "gap_path": "|".join(f"{x:.3f}" for x in gap),
                        }
                    )

        print(f"Processed batch {batch_idx}")

    out = pd.DataFrame(rows)
    if out.empty:
        print("No candidate found with current thresholds.")
        return

    out = out.sort_values(["rhs_increase", "gap_decrease", "monotonic_hits"], ascending=[False, False, False])
    out_path = BASE / "analysis" / "rhs_increase_cases.csv"
    out.to_csv(out_path, index=False)

    print(f"Candidates: {len(out)}")
    print(f"Thresholds: min_gap0={args.min_gap0}, min_rhs_inc={args.min_rhs_inc}")
    print("Top 5:")
    print(out.head(5).to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
