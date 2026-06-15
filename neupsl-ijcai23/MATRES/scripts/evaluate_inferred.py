#!/usr/bin/env python3
"""
Evaluate NeuPSL inferred predicates (REL.txt) against ground truth.

Usage: python evaluate_inferred.py <path_to_REL.txt>

REL.txt format: entity1\tentity2\tlabel\tprobability
Each (e1, e2) pair has 3 rows (one per label 0,1,2).
The predicted label = argmax over the 3 probabilities.
"""
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report

# ==================== CONFIG ====================
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TRE_ROOT = os.path.join(THIS_DIR, '..')
sys.path.append(TRE_ROOT)

CACHE_PATH = '/data/ddao/TRE/pretrained_models/Reasoning/MATRES/cache/'
LABEL_NAMES = ['BEFORE', 'AFTER', 'EQUAL']

from tre.data import create_dataloader

# ==================== LOAD GROUND TRUTH ====================
print("Loading test ground truth...", flush=True)
test_loader = create_dataloader(
    CACHE_PATH + 'test/token_cache_bert.pt',
    CACHE_PATH + 'test/spacy_cache_bert.jsonl',
    batch_size=128, num_workers=0, pin_memory=False
)
test_ds = test_loader.dataset

# Build ground truth map: (global_u, global_v) -> true_label
# Must replicate same ID assignment as generate_psl_data_v2.py
global_entity_map = {}
global_next_id = 0
ground_truth = {}

splits = ['train', 'valid', 'test']
# We need all 3 splits to compute global IDs correctly
all_datasets = []
for split in splits:
    loader = create_dataloader(
        CACHE_PATH + f'{split}/token_cache_bert.pt',
        CACHE_PATH + f'{split}/spacy_cache_bert.jsonl',
        batch_size=128, num_workers=0, pin_memory=False
    )
    all_datasets.append(loader.dataset)

for ds_idx, ds in enumerate(all_datasets):
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

        if ds_idx == 2:  # Only test set
            label = int(ds[i]['labels'])
            if label < 3:  # Skip VAGUE
                ground_truth[(u, v)] = label

print(f"Ground truth: {len(ground_truth)} test pairs (non-VAGUE)")

# ==================== PARSE INFERRED PREDICATES ====================
rel_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TRE_ROOT, 'results', 'MATRES-infer', 'inferred-predicates', 'REL.txt')
print(f"Reading inferred predicates from: {rel_path}")

# Parse REL.txt: (e1, e2) -> {label: prob}
predictions = defaultdict(dict)
with open(rel_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 4:
            e1, e2, label, prob = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            predictions[(e1, e2)][label] = prob

# ==================== EVALUATE ====================
all_preds, all_labels = [], []
matched = 0

for (u, v), true_label in ground_truth.items():
    if (u, v) in predictions:
        probs = predictions[(u, v)]
        # Only consider labels 0, 1, 2
        pred = max([0, 1, 2], key=lambda l: probs.get(l, 0.0))
        all_preds.append(pred)
        all_labels.append(true_label)
        matched += 1

print(f"Matched {matched}/{len(ground_truth)} test pairs in inferred predicates")

if len(all_preds) == 0:
    print("ERROR: No matching pairs found!")
    sys.exit(1)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean()
f1_w = f1_score(all_labels, all_preds, average='weighted')
all_labels_list = [0, 1, 2]
f1_m = f1_score(all_labels, all_preds, average='macro', labels=all_labels_list, zero_division=0)

print()
print("=" * 50)
print(f"  NeuPSL TEST RESULTS (n={len(all_preds)})")
print("=" * 50)
print(f"  Accuracy:     {acc:.6f}")
print(f"  F1-weighted:  {f1_w:.6f}")
print(f"  F1-macro:     {f1_m:.6f}")
print("=" * 50)
print()
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, labels=all_labels_list, digits=4, zero_division=0))
