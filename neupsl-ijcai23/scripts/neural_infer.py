#!/usr/bin/env python3
"""
Neural-only inference: predict using only the neural network without PSL.
No learning, no PSL reasoning - just forward pass + evaluation.

Usage: python neural_infer.py <experiment> [cache_path]
Example: python neural_infer.py MATRES
         python neural_infer.py TBD /data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/
"""
import os
import sys
import json
import datetime
import numpy as np
import torch
from transformers import AutoModel
from sklearn.metrics import f1_score, classification_report
from collections import defaultdict

# Add project paths
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_ROOT = os.path.join(THIS_DIR, '..')
sys.path.insert(0, PROJ_ROOT)

# ==================== CONFIG ====================
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else 'MATRES'
CACHE_PATH = sys.argv[2] if len(sys.argv) > 2 else None

# Map experiment to config
EXPERIMENT_CONFIG = {
    'MATRES': {
        'cache_path': '/data/ddao/TRE/pretrained_models/Reasoning/MATRES/cache/',
        'num_classes': 3,
        'label_names': ['0', '1', '2'],  # BEFORE, AFTER, EQUAL
        'skip_label': 3,  # Skip VAGUE
    },
    'TBD': {
        'cache_path': '/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache/',
        'num_classes': 5,
        'label_names': ['AFTER', 'BEFORE', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS'],
        'skip_label': 5,  # Skip VAGUE
    },
    'TDDMan': {
        'cache_path': '/data/ddao/TRE/pretrained_models/Reasoning/TDDMan/cache/',
        'num_classes': 5,
        'label_names': ['AFTER', 'BEFORE', 'INCLUDES', 'IS_INCLUDED', 'SIMULTANEOUS'],
        'skip_label': 5,  # Skip VAGUE
    },
}

if EXPERIMENT not in EXPERIMENT_CONFIG:
    print(f"ERROR: Unknown experiment: {EXPERIMENT}")
    sys.exit(1)

config = EXPERIMENT_CONFIG[EXPERIMENT]
if CACHE_PATH is not None:
    config['cache_path'] = CACHE_PATH
    if not config['cache_path'].endswith('/'):
        config['cache_path'] += '/'

RESULTS_BASE = os.path.abspath(os.path.join(PROJ_ROOT, 'results'))
RESULTS_DIR = os.path.join(RESULTS_BASE, f'{EXPERIMENT}-neural-infer')
CHECKPOINT_DIR = os.path.join(RESULTS_BASE, EXPERIMENT)
MODEL_DIR = os.path.abspath(os.path.join(PROJ_ROOT, EXPERIMENT, 'tre'))

os.makedirs(RESULTS_DIR, exist_ok=True)

# Redirect output to log file
log_path = os.path.join(RESULTS_DIR, 'neural_infer.log')
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, 'a')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = sys.stdout

print(f"\n{'='*70}")
print(f"Neural-only Inference: {EXPERIMENT}")
print(f"Time: {datetime.datetime.now()}")
print(f"{'='*70}\n")

# ==================== LOAD MODEL AND DATA ====================
print(f"[1] Loading model and data...")
print(f"    Experiment: {EXPERIMENT}")
print(f"    Num classes: {config['num_classes']}")
print(f"    Cache path: {config['cache_path']}")
print(f"    Checkpoint: {CHECKPOINT_DIR}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Device: {device}\n")

# Import model utilities
sys.path.insert(0, MODEL_DIR)
from model import BaseLine
from data import create_dataloader, tre_collate_cached

# Load datasets (train, valid, test) to build global ID map
print("[2] Loading datasets...")
splits = ['train', 'valid', 'test']
datasets = []
loaders = []
for split in splits:
    print(f"    Loading {split}...", end='', flush=True)
    loader = create_dataloader(
        config['cache_path'] + f'{split}/token_cache_bert.pt',
        config['cache_path'] + f'{split}/spacy_cache_bert.jsonl',
        batch_size=128, num_workers=2, pin_memory=False
    )
    datasets.append(loader.dataset)
    loaders.append(loader)
    print(f" Done ({len(loader.dataset)} samples)")

# Build global entity ID map (same as PSL data generation)
print("\n[3] Building global entity ID map...")
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

        # Store ground truth for test set samples with valid labels
        if ds_idx == 2:  # test set
            label = int(ds[i]['labels'])
            if config['skip_label'] is None or label != config['skip_label']:
                ground_truth[(u, v)] = label

print(f"    Total entities mapped: {global_next_id}")
print(f"    Ground truth test pairs: {len(ground_truth)}")

# ==================== LOAD NEURAL MODEL ====================
print("\n[4] Initializing neural model...")
bert = AutoModel.from_pretrained('bert-base-uncased')

# Check if embeddings need resizing
max_id = max(ds.input_ids.max().item() for ds in datasets if ds is not None)
if max_id >= bert.config.vocab_size:
    new_size = max_id + 1
    print(f"    Resizing embeddings from {bert.config.vocab_size} to {new_size}")
    bert.resize_token_embeddings(new_size)

model = BaseLine(bert, num_classes=config['num_classes'], freeze_bert=True)
model.to(device)
model.eval()

# Load checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
if os.path.exists(checkpoint_path):
    print(f"    Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"    Checkpoint loaded (step={checkpoint.get('step', '?')})")
else:
    legacy_path = os.path.join(CHECKPOINT_DIR, 'pytorch_model.bin')
    if os.path.exists(legacy_path):
        print(f"    Loading legacy weights from: {legacy_path}")
        state_dict = torch.load(legacy_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    else:
        print(f"    WARNING: No checkpoint found at {checkpoint_path}")
        print(f"    WARNING: Using untrained model!")

# ==================== NEURAL FORWARD PASS ====================
print("\n[5] Running neural forward pass on test set...")
test_loader = loaders[2]
test_ds = datasets[2]

predictions = defaultdict(dict)
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        # Move batch to device
        for key in ['input_ids', 'attention_mask', 'word_marks', 'e1_marks', 'e2_marks', 'labels', 
                   'e1_type_ids', 'e2_type_ids']:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        batch_size_actual = len(batch['doc_ids'])
        
        # Forward pass
        # e1_ids, e2_ids are lists of tensors/scalars
        e1_ids = [x.item() if hasattr(x, 'item') else x for x in batch['e1_ids']]
        e2_ids = [x.item() if hasattr(x, 'item') else x for x in batch['e2_ids']]
        entity_pairs = list(zip(e1_ids, e2_ids))
        
        out = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            word_marks=batch['word_marks'],
            e1_marks=batch['e1_marks'],
            e2_marks=batch['e2_marks'],
            entity_pairs=entity_pairs,
            doc_ids=batch['doc_ids'],
            e1_type_ids=batch.get('e1_type_ids'),
            e2_type_ids=batch.get('e2_type_ids'),
        )
        
        logits = out['LF']
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        labels_batch = batch['labels'].cpu().numpy()
        
        # Store predictions with global IDs
        for j in range(batch_size_actual):
            e1_local = e1_ids[j]
            e2_local = e2_ids[j]
            doc = batch['doc_ids'][j]
            
            split = 'test'
            try:
                u = global_entity_map[(split, str(doc), str(e1_local))]
                v = global_entity_map[(split, str(doc), str(e2_local))]
                
                pred_label = preds[j].item()
                true_label = int(labels_batch[j])
                prob = probs[j][pred_label].item()
                
                # Store for evaluation
                if (u, v) in ground_truth:
                    all_preds.append(pred_label)
                    all_labels.append(true_label)
            except KeyError:
                # Skip if not in global map
                pass

print(f"    Total predictions: {len(all_preds)}")

# ==================== EVALUATION ====================
print("\n[6] Evaluating predictions...")

if len(all_preds) == 0:
    print("    ERROR: No predictions generated!")
    sys.exit(1)

accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
f1_micro = f1_score(all_labels, all_preds, average='micro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')
f1_macro = f1_score(all_labels, all_preds, average='macro')

print(f"\n{'='*70}")
print(f"  NEURAL-ONLY TEST RESULTS (n={len(all_preds)})")
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

# ==================== SAVE RESULTS ====================
print("\n[7] Saving results...")

metrics_path = os.path.join(RESULTS_DIR, 'neural_evaluation_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write('\n' + '='*70 + '\n')
    f.write(f'  NEURAL-ONLY TEST RESULTS (n={len(all_preds)})\n')
    f.write('='*70 + '\n')
    f.write(f'  Accuracy:     {accuracy:.6f}\n')
    f.write(f'  F1-micro:     {f1_micro:.6f}\n')
    f.write(f'  F1-weighted:  {f1_weighted:.6f}\n')
    f.write(f'  F1-macro:     {f1_macro:.6f}\n')
    f.write('='*70 + '\n\n')
    f.write(classification_report(all_labels, all_preds, 
                                  target_names=config['label_names'],
                                  labels=range(config['num_classes']),
                                  digits=4, zero_division=0))

print(f"    Metrics saved to: {metrics_path}")
print(f"    Log saved to: {log_path}\n")

print(f"{'='*70}")
print(f"Neural-only inference complete!")
print(f"Results directory: {RESULTS_DIR}")
print(f"{'='*70}\n")

log_file.close()
