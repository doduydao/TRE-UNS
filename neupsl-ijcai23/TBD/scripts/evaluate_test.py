#!/usr/bin/env python3
"""Independent evaluation of the saved NeuPSL model on the TBD test set.

TBD label mapping (LabelEncoder alphabetical):
AFTER=0, BEFORE=1, INCLUDES=2, IS_INCLUDED=3, SIMULTANEOUS=4, VAGUE=5
VAGUE (label 5) is excluded from evaluation.
"""
import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModel

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TRE_ROOT = os.path.join(THIS_DIR, '..')
sys.path.append(TRE_ROOT)

from tre.model import BaseLine
from tre.data import TRECachedDataset, tre_collate_cached, create_dataloader

# ==================== CONFIG ====================
CACHE_PATH = '/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/'
CKPT_PATH = os.path.join(THIS_DIR, '..', '..', 'results', 'TBD', 'checkpoint.pt')
BATCH_SIZE = 128
NUM_CLASSES = 5
VAGUE_LABEL = 5
LABEL_NAMES = ['AFTER', 'BEFORE', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS']

# ==================== LOAD MODEL ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Loading checkpoint from: {CKPT_PATH}")

# Load test dataset
test_loader = create_dataloader(
    CACHE_PATH + 'test/token_cache_bert.pt',
    CACHE_PATH + 'test/spacy_cache_bert.jsonl',
    batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
)
test_ds = test_loader.dataset
print(f"Test dataset: {len(test_ds)} samples")

# Init model
max_id = test_ds.input_ids.max().item()
bert = AutoModel.from_pretrained('bert-base-uncased')
if max_id >= bert.config.vocab_size:
    bert.resize_token_embeddings(max_id + 1)

model = BaseLine(bert, num_classes=NUM_CLASSES, freeze_bert=True)
model.to(device)

# Load saved checkpoint
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded from step={ckpt.get('step')}, fit_call_count={ckpt.get('fit_call_count')}, best_valid_f1={ckpt.get('best_valid_f1', 'N/A')}")
else:
    model.load_state_dict(ckpt)  # Legacy format
model.eval()
print("Model loaded successfully.\n")

# ==================== EVALUATE ====================
all_preds, all_labels = [], []

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
        preds = torch.argmax(out['LF'][:, :NUM_CLASSES], dim=1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()

        for p, l in zip(preds, labels):
            if int(l) < VAGUE_LABEL:  # Skip VAGUE (label 5)
                all_preds.append(int(p))
                all_labels.append(int(l))

# ==================== RESULTS ====================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

all_labels_list = list(range(NUM_CLASSES))  # [0, 1, 2, 3, 4]

acc = (all_preds == all_labels).mean()
f1_w = f1_score(all_labels, all_preds, average='weighted')
f1_m = f1_score(all_labels, all_preds, average='macro', labels=all_labels_list, zero_division=0)
f1_micro = f1_score(all_labels, all_preds, average='micro', labels=all_labels_list, zero_division=0)

print("=" * 50)
print(f"  TEST RESULTS (n={len(all_preds)})")
print("=" * 50)
print(f"  Accuracy:     {acc:.6f}")
print(f"  F1-micro:     {f1_micro:.6f}")
print(f"  F1-weighted:  {f1_w:.6f}")
print(f"  F1-macro:     {f1_m:.6f}")
print("=" * 50)
print()
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, labels=all_labels_list, digits=4, zero_division=0))
