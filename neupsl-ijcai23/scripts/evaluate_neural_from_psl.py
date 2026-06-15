#!/usr/bin/env python3
"""
Evaluate Neural predictions extracted from NEURAL.txt (BEFORE PSL reasoning).

This script:
1. Reads NEURAL.txt from PSL inference output
2. Extracts predictions (argmax) for each (e1, e2) pair
3. Evaluates against ground truth
4. Compares with step=1 PSL results to show PSL impact

Usage: python evaluate_neural_from_psl.py <experiment> <neural_txt_path>
Example: python evaluate_neural_from_psl.py MATRES results/MATRES-infer/inferred-predicates/NEURAL.txt
"""
import os
import sys
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report

# Add project paths
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_ROOT = os.path.join(THIS_DIR, '..')
sys.path.insert(0, PROJ_ROOT)

# ==================== CONFIG ====================
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else 'MATRES'
NEURAL_TXT_PATH = sys.argv[2] if len(sys.argv) > 2 else None

EXPERIMENT_CONFIG = {
    'MATRES': {
        'cache_path': '/data/ddao/TRE/pretrained_models/Reasoning/MATRES/cache/',
        'num_classes': 3,
        'label_names': ['0', '1', '2'],
        'skip_label': 3,
    },
    'TBD': {
        'cache_path': '/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/',
        'num_classes': 5,
        'label_names': ['AFTER', 'BEFORE', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS'],
        'skip_label': 5,
    },
    'TDDMan': {
        'cache_path': '/data/ddao/TRE/pretrained_models/Reasoning/TDDMan/cache/',
        'num_classes': 5,
        'label_names': ['AFTER', 'BEFORE', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS'],
        'skip_label': 5,
    },
}

if EXPERIMENT not in EXPERIMENT_CONFIG:
    print(f"ERROR: Unknown experiment: {EXPERIMENT}")
    sys.exit(1)

config = EXPERIMENT_CONFIG[EXPERIMENT]

# Default NEURAL.txt path
if NEURAL_TXT_PATH is None:
    NEURAL_TXT_PATH = os.path.join(PROJ_ROOT, 'results', f'{EXPERIMENT}-infer', 
                                    'inferred-predicates', 'NEURAL.txt')

if not os.path.exists(NEURAL_TXT_PATH):
    print(f"ERROR: NEURAL.txt not found at: {NEURAL_TXT_PATH}")
    sys.exit(1)

print(f"{'='*70}")
print(f"Evaluating Neural Predictions from PSL Output")
print(f"Experiment: {EXPERIMENT}")
print(f"NEURAL.txt: {NEURAL_TXT_PATH}")
print(f"{'='*70}\n")

# ==================== LOAD GROUND TRUTH ====================
print("[1] Loading ground truth...")

sys.path.insert(0, os.path.join(PROJ_ROOT, EXPERIMENT, 'tre'))
from data import create_dataloader

splits = ['train', 'valid', 'test']
datasets = []
for split in splits:
    loader = create_dataloader(
        config['cache_path'] + f'{split}/token_cache_bert.pt',
        config['cache_path'] + f'{split}/spacy_cache_bert.jsonl',
        batch_size=128, num_workers=0, pin_memory=False
    )
    datasets.append(loader.dataset)

# Build global entity ID map
global_entity_map = {}
global_next_id = 0
ground_truth = {}

for ds_idx, ds in enumerate(datasets):
    split = splits[ds_idx]
    for i in range(len(ds.e1_ids)):
        e1 = ds.e1_ids[i].item() if hasattr(ds.e1_ids[i], 'item') else ds.e1_ids[i]
        e2 = ds.e2_ids[i].item() if hasattr(ds.e2_ids[i], 'item') else ds.e2_ids[i]
        doc = ds.doc_ids[i]

        for entity in [e1, e2]:
            k = (split, str(doc), str(entity))
            if k not in global_entity_map:
                global_entity_map[k] = global_next_id
                global_next_id += 1

        u = global_entity_map[(split, str(doc), str(e1))]
        v = global_entity_map[(split, str(doc), str(e2))]

        if ds_idx == 2:  # test set
            label = int(ds[i]['labels'])
            if config['skip_label'] is None or label != config['skip_label']:
                ground_truth[(u, v)] = label

print(f"Total ground truth test pairs: {len(ground_truth)}\n")

# ==================== PARSE NEURAL.TXT ====================
print("[2] Parsing NEURAL.txt...")
print(f"Reading from: {NEURAL_TXT_PATH}")

# Parse NEURAL.txt: (e1, e2) -> {label: prob}
neural_predictions = defaultdict(dict)
with open(NEURAL_TXT_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 4:
            e1, e2, label, prob = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            neural_predictions[(e1, e2)][label] = prob

print(f"Parsed {len(neural_predictions)} neural predictions")
print(f"Average labels per pair: {sum(len(p) for p in neural_predictions.values()) / len(neural_predictions):.2f}\n")

# ==================== EVALUATE NEURAL ====================
print("[3] Evaluating neural predictions...")

all_preds = []
all_labels = []
matched_count = 0

for (u, v), true_label in ground_truth.items():
    if (u, v) in neural_predictions:
        probs = neural_predictions[(u, v)]
        # Get argmax prediction across available labels
        if len(probs) > 0:
            pred = max(range(config['num_classes']), 
                      key=lambda l: probs.get(l, 0.0))
            all_preds.append(pred)
            all_labels.append(true_label)
            matched_count += 1

print(f"Matched predictions: {matched_count}/{len(ground_truth)}")

if len(all_preds) == 0:
    print("ERROR: No predictions matched!")
    sys.exit(1)

# ==================== METRICS ====================
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
f1_micro = f1_score(all_labels, all_preds, average='micro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')
f1_macro = f1_score(all_labels, all_preds, average='macro')

print(f"\n{'='*70}")
print(f"  NEURAL PREDICTIONS FROM PSL OUTPUT (n={len(all_preds)})")
print(f"{'='*70}")
print(f"  Accuracy:     {accuracy:.6f}")
print(f"  F1-micro:     {f1_micro:.6f}")
print(f"  F1-weighted:  {f1_weighted:.6f}")
print(f"  F1-macro:     {f1_macro:.6f}")
print(f"{'='*70}\n")

print(classification_report(all_labels, all_preds,
                           target_names=config['label_names'],
                           labels=range(config['num_classes']),
                           digits=4, zero_division=0))

print(f"\n{'='*70}")
print(f"INTERPRETATION:")
print(f"{'='*70}")
print(f"This shows neural predictions BEFORE PSL reasoning.")
print(f"")
print(f"Compare with:")
print(f"  1. Neural-only (neural_infer.sh): {os.path.join(PROJ_ROOT, f'results/{EXPERIMENT}-neural-infer/neural_evaluation_metrics.txt')}")
print(f"  2. PSL+Neural (infer.sh): {os.path.join(PROJ_ROOT, f'results/{EXPERIMENT}-infer/evaluation_metrics.txt')}")
print(f"")
print(f"If neural > PSL+Neural: PSL rules may be making predictions worse")
print(f"If neural < PSL+Neural: PSL rules are improving predictions")
print(f"If neural ≈ neural_infer: This script correctly extracts neural predictions")
print(f"{'='*70}\n")
