"""
Q Probability Trajectory — TBD dataset
========================================
Capture actual Q probability distributions at each reasoning step
for a specific event triplet.

Output: table + plot like:
  Step | (e1,e2)      | (e2,e3)      | (e1,e3)      | Violation
  0    | BEFORE: 0.81 | BEFORE: 0.77 | AFTER: 0.46  | cao
  1    | BEFORE: 0.82 | BEFORE: 0.79 | BEFORE: 0.41 | giảm
  ...

Usage:
  python q_trajectory.py
  python q_trajectory.py --doc CNN19980213.2130.0155 --triplet e52 e54 e57
"""
import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import DataLoader

# ── add tre_reasoner to path ──────────────────────────────────
REASONER_DIR = Path(__file__).resolve().parent.parent.parent / "tre_reasoner"
sys.path.insert(0, str(REASONER_DIR))

from data import TRECachedDataset, tre_collate_cached
from data_sampler import DocBatchSampler
from model import create_model
from conf_loader import load_runtime_config

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent                        # TBD_old/
REASONER_ROOT = SCRIPT_DIR.parent.parent               # Reasoning/

CHECKPOINT  = BASE_DIR / "checkpoint" / "TBD_baseline_reasoning_model.pt"
CONF_FILE   = REASONER_ROOT / "script" / "config" / "TBD.conf"
RELATION_MAP = {"AFTER": 0, "BEFORE": 1, "INCLUDES": 2,
                "IS_INCLUDED": 3, "SIMULTANEOUS": 4, "VAGUE": 5}
# Use rule file directly to avoid profile switching confusion.
RULES_FILE = REASONER_ROOT / "rules" / "rules_TBD_old.txt"
IDX_TO_REL  = {v: k for k, v in RELATION_MAP.items()}
NON_VAGUE_IDXS = [idx for name, idx in RELATION_MAP.items() if name != "VAGUE"]
NUM_CLASSES = 6

# Steps at which we capture Q
CAPTURE_STEPS = [0, 10, 50, 100, 500, 1000]

# Default target (best triplet found by case_study_trajectory.py)
DEFAULT_DOC     = "CNN19980213.2130.0155"
DEFAULT_TRIPLET = ("e52", "e54", "e57")    # BEFORE∘BEFORE violation resolved by t=50


# ─────────────────────────────────────────────────────────────
# Parse args
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--doc",     default=DEFAULT_DOC)
parser.add_argument("--triplet", nargs=3, default=list(DEFAULT_TRIPLET),
                    metavar=("E1", "E2", "E3"))
parser.add_argument("--gpu",     type=int, default=0)
parser.add_argument("--auto",    action="store_true",
                    help="Auto-find best triplet in target doc")
parser.add_argument("--find-clear-example", action="store_true",
                    help="Auto-find a triplet dominated by BEFORE/AFTER/SIMULTANEOUS")
parser.add_argument("--model-type", choices=["trained", "untrained"], default="trained",
                    help="Use checkpoint-trained model or random-initialized model")
parser.add_argument("--anchor-beta", type=float, default=0.15,
                    help="Blend each reasoning step with encoder P to reduce drift (0 disables)")
parser.add_argument("--max-step-delta", type=float, default=0.05,
                    help="Cap per-step probability change per class (0 disables)")
args = parser.parse_args()

TARGET_DOC     = args.doc
A, B, C        = args.triplet
DEVICE         = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

TARGET_RELS = {"BEFORE", "AFTER", "SIMULTANEOUS"}
TRANS_RULES = {
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

print("=" * 65)
print("Q Probability Trajectory Capture")
print(f"  Doc     : {TARGET_DOC}")
print(f"  Triplet : ({A}, {B}, {C})")
print(f"  Device  : {DEVICE}")
print(f"  Model   : {args.model_type}")
print(f"  Rules   : {RULES_FILE.name}")
print(f"  Steps   : {CAPTURE_STEPS}")
print(f"  Anchor  : beta={args.anchor_beta}, max_delta={args.max_step_delta}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────
# 1. Load model
# ─────────────────────────────────────────────────────────────
print("\n[1] Loading model...")
dataset_cfg, model_cfg = load_runtime_config("TBD", str(CONF_FILE))

from transformers import AutoModel, BertTokenizerFast
bert_name = model_cfg.bert_model
tokenizer  = BertTokenizerFast.from_pretrained(bert_name)
bert       = AutoModel.from_pretrained(bert_name)
special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
tokenizer.add_special_tokens(special_tokens)
bert.resize_token_embeddings(len(tokenizer))

model = create_model(
    mode="baseline_reasoning",
    bert=bert,
    num_classes=NUM_CLASSES,
    rule_file_path=str(RULES_FILE),
    relation_map=RELATION_MAP,
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

if args.model_type == "trained":
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # If rule count changed, keep old rule weights and force new rules to 1.0.
    ckpt_logits = state_dict.get("reasoning.w_rules_logits", None)
    curr_logits = model.state_dict().get("reasoning.w_rules_logits", None)
    if ckpt_logits is not None and curr_logits is not None and ckpt_logits.shape != curr_logits.shape:
        state_dict = dict(state_dict)
        n_old = int(ckpt_logits.shape[0])
        n_new = int(curr_logits.shape[0])
        merged_logits = torch.ones_like(curr_logits)
        n_copy = min(n_old, n_new)
        merged_logits[:n_copy] = ckpt_logits[:n_copy].to(merged_logits.device)
        if n_new > n_old:
            print(
                f"  WARNING: rule count changed ({n_old} -> {n_new}). "
                f"Forcing {n_new - n_old} new rule logits to 1.0 by default."
            )
        else:
            print(
                f"  WARNING: rule count changed ({n_old} -> {n_new}). "
                "Truncating checkpoint rule logits to fit current rules."
            )
        state_dict["reasoning.w_rules_logits"] = merged_logits
    model.load_state_dict(state_dict, strict=False)
model.to(DEVICE).eval()
if args.model_type == "trained":
    print(f"  Loaded: {CHECKPOINT.name}")
else:
    print("  Loaded: random initialization (no checkpoint)")

reasoning = model.reasoning  # UnifiedNeuralReasoningLayerV2


# ─────────────────────────────────────────────────────────────
# 2. Load test data — find the target document batch
# ─────────────────────────────────────────────────────────────
print("\n[2] Loading test data...")
dataset = TRECachedDataset(dataset_cfg.test_token_path, dataset_cfg.test_jsonl_path)

sampler    = DocBatchSampler(dataset.doc_ids, batch_size=256, shuffle=False)
dataloader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=tre_collate_cached)

target_batch = None
for batch in dataloader:
    doc_ids_batch = batch.get("doc_ids", [])
    if isinstance(doc_ids_batch, torch.Tensor):
        doc_ids_batch = doc_ids_batch.tolist()
    if TARGET_DOC in doc_ids_batch:
        target_batch = batch
        n_doc_pairs = sum(1 for d in doc_ids_batch if d == TARGET_DOC)
        print(f"  Found doc '{TARGET_DOC}' — {n_doc_pairs} pairs (batch size={len(doc_ids_batch)})")
        break

if target_batch is None:
    raise RuntimeError(f"Document '{TARGET_DOC}' not found in test set.")


# ─────────────────────────────────────────────────────────────
# 3. Extract P from encoder (no reasoning)
# ─────────────────────────────────────────────────────────────
print("\n[3] Running encoder...")

def to_dev(x):
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE)
    return x

def take_rows(x, indices):
    if isinstance(x, torch.Tensor):
        idx = torch.tensor(indices, dtype=torch.long, device=x.device)
        return x.index_select(0, idx)
    if isinstance(x, list):
        return [x[i] for i in indices]
    return x

doc_ids_raw = target_batch.get("doc_ids", [])
if isinstance(doc_ids_raw, torch.Tensor):
    doc_ids_raw = doc_ids_raw.tolist()
target_doc_indices = [i for i, d in enumerate(doc_ids_raw) if d == TARGET_DOC]
if not target_doc_indices:
    raise RuntimeError(f"No pairs found for target doc '{TARGET_DOC}' in selected batch.")
print(f"  Using only target-doc rows for encoder forward: {len(target_doc_indices)} pairs")

input_ids      = to_dev(take_rows(target_batch["input_ids"], target_doc_indices))
attention_mask = to_dev(take_rows(target_batch["attention_mask"], target_doc_indices))
word_marks     = to_dev(take_rows(target_batch.get("word_marks"), target_doc_indices))
e1_marks       = to_dev(take_rows(target_batch.get("e1_marks"), target_doc_indices))
e2_marks       = to_dev(take_rows(target_batch.get("e2_marks"), target_doc_indices))
e1_type_ids    = to_dev(take_rows(target_batch.get("e1_type_ids"), target_doc_indices))
e2_type_ids    = to_dev(take_rows(target_batch.get("e2_type_ids"), target_doc_indices))
labels         = take_rows(target_batch.get("labels"), target_doc_indices)

e1_ids_all = take_rows(target_batch.get("e1_ids", []), target_doc_indices)
e2_ids_all = take_rows(target_batch.get("e2_ids", []), target_doc_indices)
doc_ids_all = take_rows(doc_ids_raw, target_doc_indices)

# Build entity_pairs list (all docs in batch, used by reasoner)
entity_pairs = list(zip(e1_ids_all, e2_ids_all))
# Build pair index for target doc
pair_to_idx = {(e1_ids_all[i], e2_ids_all[i]): i for i in range(len(doc_ids_all))}

# Check triplet pairs exist
pair_AB = (A, B)
pair_BC = (B, C)
pair_AC = (A, C)
for p in [pair_AB, pair_BC, pair_AC]:
    if p not in pair_to_idx:
        doc_e1s = sorted({e1_ids_all[i] for i in range(len(doc_ids_all))})
        print(f"  WARNING: pair {p} not found in doc {TARGET_DOC}. Available e1s: {doc_e1s[:10]}")

with torch.no_grad():
    nn_out = model.neural(input_ids, attention_mask, word_marks, e1_marks, e2_marks)

P = nn_out["P"].to(DEVICE)  # [N, num_classes], float on device
print(f"  P shape: {P.shape}  (N_pairs={P.shape[0]}, num_classes={P.shape[1]})")

# Ground truth labels
if labels is not None:
    gt_labels = labels
    for pair, name in [(pair_AB, f"({A},{B})"), (pair_BC, f"({B},{C})"), (pair_AC, f"({A},{C})")]:
        idx = pair_to_idx.get(pair)
        if idx is not None and gt_labels is not None:
            gt_idx = gt_labels[idx].item() if isinstance(gt_labels, torch.Tensor) else gt_labels[idx]
            print(f"  GT {name} = {IDX_TO_REL.get(gt_idx, gt_idx)}")


# ─────────────────────────────────────────────────────────────
# 4. Custom step-by-step reasoning loop (captures Q at each step)
# ─────────────────────────────────────────────────────────────
print(f"\n[4] Running custom reasoning loop (max {max(CAPTURE_STEPS)} steps)...")

# Build context (precompute rule structures)
ctx, pre_batch_data = reasoning._prepare_context(
    P, entity_pairs, doc_ids_all, e1_type_ids, e2_type_ids
)

MAX_CAPTURE = max(CAPTURE_STEPS)
capture_set = set(CAPTURE_STEPS)
Q_snapshots = {}   # step → tensor [N, num_classes] on CPU

# Step 0: encoder output (no reasoning applied yet)
Q_snapshots[0] = P.detach().cpu().clone()

Q = P.clone().detach().to(DEVICE)

step_size  = reasoning.step_size
clip_val   = getattr(reasoning, "grad_clip_value", 10.0)
anchor_beta = max(0.0, float(args.anchor_beta))
max_step_delta = max(0.0, float(args.max_step_delta))
eps = 1e-12

for step in range(1, MAX_CAPTURE + 1):
    with torch.enable_grad():
        Q_req = Q.detach().requires_grad_(True)
        res   = reasoning._compute_reason_energy(P, Q_req, ctx, pre_batch_data)
        E_re  = res["E_re"]
        g_q,  = torch.autograd.grad(E_re, Q_req, create_graph=False)
        g_q   = torch.clamp(g_q, -clip_val, clip_val)
        log_Q = torch.log(Q + eps) - step_size * g_q
        Q_new = torch.softmax(log_Q, dim=-1)

        # Safety 1: keep Q close to encoder belief to avoid wrong-label drift.
        if anchor_beta > 0.0:
            Q_new = (1.0 - anchor_beta) * Q_new + anchor_beta * P

        # Safety 2: prevent abrupt per-step jumps when reducing VAGUE mass.
        if max_step_delta > 0.0:
            delta = torch.clamp(Q_new - Q, min=-max_step_delta, max=max_step_delta)
            Q_new = Q + delta

        Q_new = torch.clamp(Q_new, min=eps)
        Q = (Q_new / Q_new.sum(dim=-1, keepdim=True)).detach()

    if step in capture_set:
        Q_snapshots[step] = Q.cpu().clone()
        E_val = res["E_re"].item()
        vc    = res.get("VC", "?")
        print(f"  t={step:>4}  E={E_val:.4f}  VC={vc}")

print("  Done.")


# ─────────────────────────────────────────────────────────────
# 5. Extract per-pair probabilities
# ─────────────────────────────────────────────────────────────
def get_probs(step, pair):
    idx = pair_to_idx.get(pair)
    if idx is None:
        return None
    return Q_snapshots[step][idx].numpy()   # [num_classes]


def top_non_vague_label_prob_from_probs(probs):
    if probs is None:
        return "—", 0.0, None
    local_idx = int(np.argmax(probs[NON_VAGUE_IDXS]))
    top_idx = NON_VAGUE_IDXS[local_idx]
    return IDX_TO_REL[top_idx], float(probs[top_idx]), top_idx


def top_label_prob(step, pair):
    """Returns the top non-VAGUE label and its original probability."""
    probs = get_probs(step, pair)
    label, prob, _ = top_non_vague_label_prob_from_probs(probs)
    return label, prob


def relation_prob(step, pair, rel_name):
    probs = get_probs(step, pair)
    if probs is None:
        return 0.0
    return float(probs[RELATION_MAP[rel_name]])


def find_clear_example_triplet():
    """Find a triplet whose trajectory is dominated by BEFORE/AFTER/SIMULTANEOUS."""
    successors = {}
    for (e1, e2) in pair_to_idx.keys():
        successors.setdefault(e1, []).append(e2)

    steps_core = [0, 1, 5, 10, 50, 100]
    candidates = []

    for (a, b) in pair_to_idx.keys():
        for c in successors.get(b, []):
            if a == c:
                continue
            if (a, c) not in pair_to_idx:
                continue

            rel_ab = [top_label_prob(s, (a, b))[0] for s in steps_core]
            rel_bc = [top_label_prob(s, (b, c))[0] for s in steps_core]
            rel_ac = [top_label_prob(s, (a, c))[0] for s in steps_core]
            label_set = set(rel_ab + rel_bc + rel_ac)
            if not ({"AFTER", "SIMULTANEOUS"} & label_set):
                continue

            # Prefer trajectories with many BEFORE/AFTER/SIMULTANEOUS labels
            target_count = sum(1 for x in (rel_ab + rel_bc + rel_ac) if x in TARGET_RELS)
            target_ratio = target_count / 18.0  # 18 = 3 pairs * 6 steps

            severities = []
            valid_steps = 0
            for s, r1, r2 in zip(steps_core, rel_ab, rel_bc):
                expected = TRANS_RULES.get((r1, r2))
                if expected is None or expected not in TARGET_RELS:
                    continue
                valid_steps += 1
                severities.append(1.0 - relation_prob(s, (a, c), expected))

            if valid_steps < 3:
                continue

            improve = severities[0] - severities[-1]
            decreases = sum(1 for i in range(1, len(severities)) if severities[i] <= severities[i - 1])

            # Need at least some decreasing inconsistency + non-trivial violation initially
            if severities[0] < 0.30:
                continue
            if improve < 0.08:
                continue

            score = improve + 0.08 * decreases + 0.6 * target_ratio
            candidates.append((score, (a, b, c), severities, rel_ab, rel_bc, rel_ac))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def find_increase_example_triplet():
    """Find triplet where expected relation prob on (A,C) increases from t=0 to t=50.

    Expected relation is fixed from step-0 top labels of (A,B) and (B,C).
    """
    successors = {}
    for (e1, e2) in pair_to_idx.keys():
        successors.setdefault(e1, []).append(e2)

    candidates = []
    for (a, b) in pair_to_idx.keys():
        for c in successors.get(b, []):
            if a == c or (a, c) not in pair_to_idx:
                continue

            r_ab0 = top_label_prob(0, (a, b))[0]
            r_bc0 = top_label_prob(0, (b, c))[0]
            expected = TRANS_RULES.get((r_ab0, r_bc0))
            if expected is None or expected not in TARGET_RELS:
                continue

            # Keep examples where AB/BC are mostly temporal labels in early steps
            early_steps = [0, 1, 5, 10, 50]
            rel_ab = [top_label_prob(s, (a, b))[0] for s in early_steps]
            rel_bc = [top_label_prob(s, (b, c))[0] for s in early_steps]
            temporal_ratio = (
                sum(1 for x in rel_ab if x in TARGET_RELS)
                + sum(1 for x in rel_bc if x in TARGET_RELS)
            ) / 10.0
            if temporal_ratio < 0.8:
                continue

            p0 = relation_prob(0, (a, c), expected)
            p1 = relation_prob(1, (a, c), expected)
            p5 = relation_prob(5, (a, c), expected)
            p10 = relation_prob(10, (a, c), expected)
            p50 = relation_prob(50, (a, c), expected)

            # Main objective: expected probability should trend upward in early steps
            inc_0_50 = p50 - p0
            inc_0_10 = p10 - p0
            monotonic_hits = sum([
                p1 >= p0,
                p5 >= p1,
                p10 >= p5,
                p50 >= p10,
            ])

            if inc_0_50 < 0.03 and inc_0_10 < 0.02:
                continue

            # Penalize VAGUE dominance on AC in early steps
            vague0 = relation_prob(0, (a, c), "VAGUE")
            vague50 = relation_prob(50, (a, c), "VAGUE")
            vague_penalty = 0.4 * (vague0 + vague50)

            score = inc_0_50 + 0.5 * inc_0_10 + 0.02 * monotonic_hits + 0.1 * temporal_ratio - vague_penalty
            candidates.append((
                score, (a, b, c), expected,
                [p0, p1, p5, p10, p50],
                rel_ab, rel_bc,
            ))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def find_dominant_temporal_triplet():
    """Fallback: choose triplet with strongest BEFORE/AFTER/SIMULTANEOUS dominance."""
    successors = {}
    for (e1, e2) in pair_to_idx.keys():
        successors.setdefault(e1, []).append(e2)

    steps_core = [0, 1, 5, 10, 50, 100]
    best = None

    for (a, b) in pair_to_idx.keys():
        for c in successors.get(b, []):
            if a == c or (a, c) not in pair_to_idx:
                continue
            rel_ab = [top_label_prob(s, (a, b))[0] for s in steps_core]
            rel_bc = [top_label_prob(s, (b, c))[0] for s in steps_core]
            rel_ac = [top_label_prob(s, (a, c))[0] for s in steps_core]
            label_set = set(rel_ab + rel_bc + rel_ac)
            if not ({"AFTER", "SIMULTANEOUS"} & label_set):
                continue
            target_count = sum(1 for x in (rel_ab + rel_bc + rel_ac) if x in TARGET_RELS)
            target_ratio = target_count / 18.0

            severities = []
            for s, r1, r2 in zip(steps_core, rel_ab, rel_bc):
                expected = TRANS_RULES.get((r1, r2))
                if expected is None:
                    continue
                severities.append(1.0 - relation_prob(s, (a, c), expected))
            if not severities:
                continue

            score = target_ratio + 0.5 * severities[0]
            cand = (score, (a, b, c), severities, rel_ab, rel_bc, rel_ac)
            if best is None or cand[0] > best[0]:
                best = cand

    return best


def violation_score(step):
    r_ab, _ = top_label_prob(step, pair_AB)
    r_bc, _ = top_label_prob(step, pair_BC)
    expected = TRANS_RULES.get((r_ab, r_bc))
    if expected is None:
        return 0.0, expected
    return 1.0 - relation_prob(step, pair_AC, expected), expected


def violation_level(step):
    score, expected = violation_score(step)
    if expected is None:
        return "-"
    if score >= 0.55:
        return "cao"
    if score >= 0.35:
        return "giảm"
    if score >= 0.18:
        return "thấp"
    return "gần nhất quán"


def is_violation(step):
    """BEFORE ∘ BEFORE → BEFORE: check if pred(AB)=BEFORE, pred(BC)=BEFORE but pred(AC)≠BEFORE."""
    def top(pair):
        p = get_probs(step, pair)
        return top_non_vague_label_prob_from_probs(p)[0]
    r_ab, r_bc, r_ac = top(pair_AB), top(pair_BC), top(pair_AC)
    # Simple: any pair from AC that contradicts transitivity
    expected = TRANS_RULES.get((r_ab, r_bc))
    if expected is None:
        return False
    return r_ac != expected


# ─────────────────────────────────────────────────────────────
# 6. Print table
# ─────────────────────────────────────────────────────────────
def fmt_cell(step, pair):
    lbl, prob = top_label_prob(step, pair)
    return f"{lbl.title()}: {prob:.2f}"


if args.find_clear_example:
    best_inc = find_increase_example_triplet()
    if best_inc is not None:
        _, triplet, expected_rel, pvals, rel_ab, rel_bc = best_inc
        A, B, C = triplet
        pair_AB, pair_BC, pair_AC = (A, B), (B, C), (A, C)
        print("\n[auto] Selected increase-focused example triplet:")
        print(f"  Triplet: ({A}, {B}, {C})")
        print(f"  Expected (from t=0): {expected_rel}")
        print(f"  P_expected((A,C)) at t=[0,1,5,10,50]: {[round(x, 3) for x in pvals]}")
        print(f"  AB top labels @0,1,5,10,50: {rel_ab}")
        print(f"  BC top labels @0,1,5,10,50: {rel_bc}")
        best = None
    else:
        best = find_clear_example_triplet()
        if best is None:
            best = find_dominant_temporal_triplet()
        if best is None:
            print("\n[auto] Không tìm được triplet phù hợp trong document này. Dùng triplet truyền vào.")
        else:
            print("\n[auto] Không có ví dụ 'clear' tuyệt đối, dùng ví dụ temporal-dominant tốt nhất.")

    if best is not None:
        _, triplet, severities, rel_ab, rel_bc, rel_ac = best
        A, B, C = triplet
        pair_AB, pair_BC, pair_AC = (A, B), (B, C), (A, C)
        print("\n[auto] Selected clear example triplet:")
        print(f"  Triplet: ({A}, {B}, {C})")
        print(f"  AB top labels @0..100: {rel_ab}")
        print(f"  BC top labels @0..100: {rel_bc}")
        print(f"  AC top labels @0..100: {rel_ac}")
        print(f"  Violation severity @0..100: {[round(x, 3) for x in severities]}")


print("\n" + "=" * 80)
print(f"Q Probability Table — doc={TARGET_DOC}  triplet=({A},{B},{C})")
print("=" * 80)
hdr = (f"{'Step':<5} | "
       f"{f'({A},{B})':<18} | "
       f"{f'({B},{C})':<18} | "
       f"{f'({A},{C})':<18} | Violation")
print(hdr)
print("-" * len(hdr))
for s in CAPTURE_STEPS:
    viol_str = violation_level(s)
    row = (f"{s:<5} | "
           f"{fmt_cell(s, pair_AB):<18} | "
           f"{fmt_cell(s, pair_BC):<18} | "
           f"{fmt_cell(s, pair_AC):<18} | {viol_str}")
    print(row)

# Also show GT
gt_lines = []
print()
for pair, name in [(pair_AB, f"({A},{B})"), (pair_BC, f"({B},{C})"), (pair_AC, f"({A},{C})")]:
    idx = pair_to_idx.get(pair)
    if idx is not None and labels is not None:
        gt_raw = labels[idx].item() if isinstance(labels, torch.Tensor) else labels[idx]
        gt_rel = IDX_TO_REL.get(gt_raw, gt_raw)
        gt_lines.append(f"GT {name} = {gt_rel}")
        print(f"  GT {name} = {gt_rel}")


# ─────────────────────────────────────────────────────────────
# 7. Probability trajectory plot
# ─────────────────────────────────────────────────────────────
print("\n[5] Generating plot...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle(
    f"Q Probability Trajectory [{args.model_type}] — ({A}, {B}, {C})   |   Doc: {TARGET_DOC}",
    fontsize=12, fontweight="bold")
if gt_lines:
    fig.text(
        0.5,
        0.93,
        "   |   ".join(gt_lines),
        ha="center",
        va="center",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

x_pos   = np.arange(len(CAPTURE_STEPS))
x_ticks = [f"t={s}" for s in CAPTURE_STEPS]

REL_COLORS = {
    "BEFORE":       "#2980b9",
    "AFTER":        "#e74c3c",
    "SIMULTANEOUS": "#27ae60",
    "INCLUDES":     "#8e44ad",
    "IS_INCLUDED":  "#e67e22",
    "VAGUE":        "#95a5a6",
}

all_pairs  = [(pair_AB, f"({A},{B})"), (pair_BC, f"({B},{C})"), (pair_AC, f"({A},{C})")]
for ax_i, (pair, pair_name) in enumerate(all_pairs):
    ax = axes[ax_i]

    # Extract probability matrix: [n_steps × num_classes]
    prob_mat = np.zeros((len(CAPTURE_STEPS), NUM_CLASSES))
    for si, s in enumerate(CAPTURE_STEPS):
        p = get_probs(s, pair)
        if p is not None:
            prob_mat[si] = p

    # Plot each relation class
    for rel_idx in range(NUM_CLASSES):
        rel_name = IDX_TO_REL[rel_idx]
        probs_r  = prob_mat[:, rel_idx]
        max_p    = probs_r.max()
        col = REL_COLORS.get(rel_name, "#888")
        lw  = 2.8 if max_p > 0.25 else 1.4
        alpha = 1.0 if max_p > 0.25 else 0.55
        ax.plot(x_pos, probs_r, "o-", color=col, linewidth=lw, alpha=alpha,
                markersize=6, label=rel_name)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks, fontsize=8.5, rotation=20)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Pair {pair_name}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability" if ax_i == 0 else "")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate top class at each step
    for xi, s in enumerate(CAPTURE_STEPS):
        p = get_probs(s, pair)
        if p is not None:
            _, top_prob, top_i = top_non_vague_label_prob_from_probs(p)
            ax.annotate(f"{top_prob:.2f}",
                        (xi, top_prob),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color=REL_COLORS.get(IDX_TO_REL[top_i], "#333"),
                        fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.88], pad=2.5)
name_suffix = args.model_type
step_suffix = "_".join(str(step) for step in CAPTURE_STEPS)
safe_doc = TARGET_DOC.replace("/", "_")
triplet_suffix = f"{A}_{B}_{C}"
out_png = SCRIPT_DIR / f"q_trajectory_{name_suffix}_{safe_doc}_{triplet_suffix}_steps_{step_suffix}.png"
out_pdf = SCRIPT_DIR / f"q_trajectory_{name_suffix}_{safe_doc}_{triplet_suffix}_steps_{step_suffix}.pdf"
plt.savefig(out_png, bbox_inches="tight", dpi=180)
plt.savefig(out_pdf, bbox_inches="tight")
print(f"  Saved: {out_png}")
print(f"  Saved: {out_pdf}")


# ─────────────────────────────────────────────────────────────
# 8. Save CSV
# ─────────────────────────────────────────────────────────────
rows = []
for s in CAPTURE_STEPS:
    row = {"step": s}
    for pair, pname in all_pairs:
        probs = get_probs(s, pair)
        if probs is not None:
            for ri in range(NUM_CLASSES):
                row[f"{pname}_{IDX_TO_REL[ri]}"] = round(float(probs[ri]), 4)
        row[f"{pname}_top"] = top_label_prob(s, pair)[0]
        row[f"{pname}_topP"] = round(top_label_prob(s, pair)[1], 4)
    row["violation"] = is_violation(s)
    rows.append(row)

df = pd.DataFrame(rows)
out_csv = SCRIPT_DIR / f"q_trajectory_{name_suffix}_steps_{step_suffix}.csv"
out_csv = SCRIPT_DIR / f"q_trajectory_{name_suffix}_{safe_doc}_{triplet_suffix}_steps_{step_suffix}.csv"
df.to_csv(out_csv, index=False)
print(f"  CSV saved: {out_csv}")

# Compact table for paper-like presentation
compact_rows = []
for s in CAPTURE_STEPS:
    compact_rows.append({
        "Step": s,
        f"({A},{B})": fmt_cell(s, pair_AB),
        f"({B},{C})": fmt_cell(s, pair_BC),
        f"({A},{C})": fmt_cell(s, pair_AC),
        "Violation": violation_level(s),
    })
compact_df = pd.DataFrame(compact_rows)
compact_csv = SCRIPT_DIR / f"q_trajectory_compact_{name_suffix}_{safe_doc}_{triplet_suffix}_steps_{step_suffix}.csv"
compact_df.to_csv(compact_csv, index=False)
print(f"  Compact CSV saved: {compact_csv}")

print("\nDone.")
