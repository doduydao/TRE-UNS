#!/usr/bin/env python3
"""
Generate a CSV report comparing Neural Baseline and NeuPSL predictions on TBD test set.
Format: doc_id,e1_id,e2_id,ground_truth,encoder_prediction,prediction
"""
import os
import sys
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModel
import csv

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TRE_ROOT = os.path.join(THIS_DIR, '..')
sys.path.append(TRE_ROOT)

from tre.model import BaseLine
from tre.data import create_dataloader, tre_collate_cached

# ==================== CONFIG ====================
CACHE_PATH = '/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/'
CKPT_PATH = os.path.join(THIS_DIR, '..', '..', 'results', 'TBD', 'checkpoint.pt')
REL_PATH = os.path.join(THIS_DIR, '..', '..', 'results', 'TBD-infer', 'inferred-predicates', 'REL.txt')
OUTPUT_CSV = os.path.join(THIS_DIR, '..', '..', 'results', 'TBD-infer', 'NeuPSL_prediction.csv')

BATCH_SIZE = 128
NUM_CLASSES = 5
VAGUE_LABEL = 5

LABEL_TO_NAME = {
    0: 'AFTER',
    1: 'BEFORE',
    2: 'INCLUDES',
    3: 'IS_INCLUDED',
    4: 'SIMULTANEOUS',
    5: 'VAGUE'
}

# ==================== 1. BUILD GLOBAL IDS CACHE ====================
print("Building global ID maps...")
global_entity_map = {}
global_next_id = 0

splits = ['train', 'valid', 'test']
all_datasets = []
for split in splits:
    loader = create_dataloader(
        os.path.join(CACHE_PATH, f'{split}/token_cache_bert.pt'),
        os.path.join(CACHE_PATH, f'{split}/spacy_cache_bert.jsonl'),
        batch_size=BATCH_SIZE, num_workers=0, pin_memory=False
    )
    all_datasets.append(loader.dataset)

test_ds = all_datasets[2]  # Test dataset
print(f"Test dataset: {len(test_ds)} samples")

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

# ==================== 2. LOAD PSL PREDICATES ====================
print(f"Reading inferred predicates from: {REL_PATH}")
psl_predictions = defaultdict(dict)
try:
    with open(REL_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                e1, e2, label, prob = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                psl_predictions[(e1, e2)][label] = prob
except FileNotFoundError:
    print(f"WARNING: No REL.txt found at {REL_PATH}. PSL predictions will be missing.")

# ==================== 3. LOAD NEURAL MODEL ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading checkpoint from: {CKPT_PATH}")

max_id = max(ds.input_ids.max().item() for ds in all_datasets)
bert = AutoModel.from_pretrained('bert-base-uncased')
if max_id >= bert.config.vocab_size:
    bert.resize_token_embeddings(max_id + 1)

model = BaseLine(bert, num_classes=NUM_CLASSES, freeze_bert=True)
model.to(device)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded successfully.")
else:
    print(f"WARNING: Checkpoint missing at {CKPT_PATH}")

# ==================== 4. GENERATE CSV ====================
print(f"Generating CSV: {OUTPUT_CSV}")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['doc_id', 'e1_id', 'e2_id', 'ground_truth', 'encoder_prediction', 'prediction'])

    with torch.no_grad():
        for start in range(0, len(test_ds), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(test_ds))
            batch_samples = [test_ds[i] for i in range(start, end)]
            batch = tre_collate_cached(batch_samples)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            entity_pairs = list(zip(batch['e1_ids'], batch['e2_ids']))
            out = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                word_marks=batch['word_marks'],
                e1_marks=batch['e1_marks'],
                e2_marks=batch['e2_marks'],
                entity_pairs=entity_pairs,
                e1_type_ids=batch.get('e1_type_ids'),
                e2_type_ids=batch.get('e2_type_ids'),
                doc_ids=batch.get('doc_ids'),
            )
            enc_preds = torch.argmax(out['LF'][:, :NUM_CLASSES], dim=1).cpu().numpy()
            true_labels = batch['labels'].cpu().numpy()

            for idx in range(len(batch_samples)):
                doc_id = batch['doc_ids'][idx]
                e1_raw = batch['e1_ids'][idx]
                e2_raw = batch['e2_ids'][idx]
                if hasattr(e1_raw, 'item'): e1_raw = e1_raw.item()
                if hasattr(e2_raw, 'item'): e2_raw = e2_raw.item()
                
                true_lbl = int(true_labels[idx])
                enc_pred_lbl = int(enc_preds[idx])

                u = global_entity_map[('test', str(doc_id), str(e1_raw))]
                v = global_entity_map[('test', str(doc_id), str(e2_raw))]

                # Get PSL prediction
                if (u, v) in psl_predictions:
                    probs = psl_predictions[(u, v)]
                    psl_pred_lbl = max(list(range(NUM_CLASSES)), key=lambda l: probs.get(l, 0.0))
                    psl_pred_str = LABEL_TO_NAME.get(psl_pred_lbl, "UNKNOWN")
                else:
                    psl_pred_str = "" # If not found

                writer.writerow([
                    str(doc_id),
                    str(e1_raw),
                    str(e2_raw),
                    LABEL_TO_NAME.get(true_lbl, str(true_lbl)),
                    LABEL_TO_NAME.get(enc_pred_lbl, str(enc_pred_lbl)),
                    psl_pred_str
                ])

print("CSV generation complete!")
