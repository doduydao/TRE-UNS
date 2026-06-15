import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRE_REASONER_DIR = os.path.join(ROOT, "tre_reasoner")
if TRE_REASONER_DIR not in sys.path:
    sys.path.insert(0, TRE_REASONER_DIR)

from conf_loader import load_runtime_config
from data import create_dataloader
from reasoning_v2 import parse_rule_line_v2, UnifiedNeuralReasoningLayerV2


def build_reasoner(dataset_cfg, model_cfg):
    with open(dataset_cfg.rule_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rules = [parse_rule_line_v2(line) for line in lines if parse_rule_line_v2(line)]
    reasoner = UnifiedNeuralReasoningLayerV2(
        rule_templates=rules,
        num_relations=dataset_cfg.num_classes,
        num_types=32,
        step_size=model_cfg.step_size,
        lambda_kl=model_cfg.lambda_kl,
        smooth_tau=model_cfg.smooth_tau,
        max_steps=model_cfg.max_steps,
        tol=model_cfg.tol,
        relation_to_index=dataset_cfg.relation_map,
        use_deq=False,
        learn_rule_weights=False,
        rule_chunk_size=model_cfg.rule_chunk_size,
    )
    reasoner.eval()
    return reasoner


def get_split_paths(dataset_cfg, split):
    split = split.lower()
    if split == "train":
        return dataset_cfg.train_token_path, dataset_cfg.train_jsonl_path
    if split in ("val", "valid", "validation"):
        return dataset_cfg.val_token_path, dataset_cfg.val_jsonl_path
    if split == "test":
        return dataset_cfg.test_token_path, dataset_cfg.test_jsonl_path
    raise ValueError(f"Unsupported split: {split}")


def run_split(reasoner, dataset_cfg, token_path, jsonl_path, batch_size):
    loader = create_dataloader(
        token_cache_path=token_path,
        spacy_jsonl_path=jsonl_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    agg = defaultdict(lambda: {"groundings": 0.0, "violations": 0.0})
    total_pairs = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].long()
            if labels.numel() == 0:
                continue

            valid = (labels >= 0) & (labels < dataset_cfg.num_classes)
            if not torch.any(valid):
                continue

            total_pairs += int(valid.sum().item())
            labels = labels.clamp(min=0, max=dataset_cfg.num_classes - 1)
            q = F.one_hot(labels, num_classes=dataset_cfg.num_classes).float()

            e1_ids = batch.get("e1_ids", [])
            e2_ids = batch.get("e2_ids", [])
            entity_pairs = list(zip(e1_ids, e2_ids)) if e1_ids and e2_ids else []
            doc_ids = batch.get("doc_ids", [])

            ctx = reasoner.build_context(q, entity_pairs, doc_ids, e1_types=None, e2_types=None)
            if ctx.max_ent == 0:
                continue

            pre_data = None if reasoner.rule_chunk_size else reasoner._precompute_batch_rules(None, ctx.T_mask, ctx)
            out = reasoner._compute_rule_energy(q, ctx, pre_batch_data=pre_data)

            for rule_name, detail in out["rule_details"].items():
                agg[rule_name]["groundings"] += float(detail.get("grounding_count", 0.0))
                agg[rule_name]["violations"] += float(detail.get("violation_count", 0.0))

    rows = []
    total_g = 0.0
    total_v = 0.0
    for rule_name, stats in agg.items():
        g = stats["groundings"]
        v = stats["violations"]
        s = max(g - v, 0.0)
        sat_rate = (s / g) if g > 0 else 0.0
        rows.append((rule_name, g, v, s, sat_rate))
        total_g += g
        total_v += v

    rows.sort(key=lambda x: x[0])
    total_s = max(total_g - total_v, 0.0)
    total_sat_rate = (total_s / total_g) if total_g > 0 else 0.0

    return {
        "rows": rows,
        "total_pairs": total_pairs,
        "total_groundings": total_g,
        "total_violations": total_v,
        "total_satisfied": total_s,
        "total_satisfaction_rate": total_sat_rate,
    }


def print_report(split, report):
    print(f"===== Split: {split} =====")
    print(f"Pairs: {report['total_pairs']}")
    print(f"Groundings: {report['total_groundings']:.0f}")
    print(f"Violations: {report['total_violations']:.0f}")
    print(f"Satisfied: {report['total_satisfied']:.0f}")
    print(f"SatisfactionRate: {report['total_satisfaction_rate']:.6f}")
    print("\nPer-rule:")
    print("Rule\tGroundings\tViolations\tSatisfied\tSatRate")
    for name, g, v, s, sat_rate in report["rows"]:
        print(f"{name}\t{g:.0f}\t{v:.0f}\t{s:.0f}\t{sat_rate:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Compute TBD rule satisfaction counts from cached splits.")
    parser.add_argument("--dataset", default="TBD", help="Dataset config name")
    parser.add_argument("--conf", default=os.path.join(ROOT, "script", "config", "TBD.conf"), help="Path to .conf")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Split to evaluate")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    dataset_cfg, model_cfg = load_runtime_config(args.dataset, conf_path=args.conf)
    reasoner = build_reasoner(dataset_cfg, model_cfg)

    splits = [args.split] if args.split != "all" else ["train", "val", "test"]
    for split in splits:
        token_path, jsonl_path = get_split_paths(dataset_cfg, split)
        report = run_split(reasoner, dataset_cfg, token_path, jsonl_path, args.batch_size)
        print_report(split, report)
        print("")


if __name__ == "__main__":
    main()
