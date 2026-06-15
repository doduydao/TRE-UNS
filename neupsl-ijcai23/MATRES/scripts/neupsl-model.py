#!/usr/bin/env python3
import os
import sys
import datetime
import time
import numpy as np
import torch
from transformers import AutoModel
from sklearn.metrics import f1_score

# Force logging to a file in APPEND mode, co-located with out.txt
_log_dir = os.environ.get('NEUPSL_LOG_DIR', '/tmp')
_log_path = os.path.join(_log_dir, 'neupsl_python.log')
sys.stdout = open(_log_path, 'a', buffering=1)
sys.stderr = sys.stdout

print(f"\n{'='*20} {datetime.datetime.now()} {'='*20}", flush=True)
print(f"Log file: {_log_path}", flush=True)
print("=== DEEP PREDICATE PYTHON SERVER STARTING ===", flush=True)

def ensure_native_byteorder(arr):
    if arr.dtype.byteorder not in ('=', '|'):
        return arr.astype(arr.dtype.newbyteorder('='))
    return arr

# Add the 'tre' module path
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TRE_ROOT = os.path.join(THIS_DIR, '..')
sys.path.append(TRE_ROOT)

import pslpython.deeppsl.model
from tre.model import BaseLine
from tre.data import TRECachedDataset, tre_collate_cached, create_dataloader


class TREModel(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._model = None
        self._optimizer = None
        self._id_map = None
        self._step = 0
        self._fit_call_count = 0
        self._last_fit_end = None
        self._best_valid_f1 = 0.0
        self._save_path = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._application = None
        self._datasets = []  # [train, valid, test]

    # ==================== HELPER METHODS ====================

    def _map_psl_data(self, data):
        """Map PSL float array rows to (original_idx, ds_idx, sample_idx)."""
        samples = []
        for i, row in enumerate(data):
            if hasattr(row, '__len__') and len(row) >= 2:
                key = (int(float(row[0])), int(float(row[1])))
                if key in self._id_map:
                    ds_idx, s_idx = self._id_map[key]
                    samples.append((i, ds_idx, s_idx))
        return samples

    def _forward_batch(self, batch_samples):
        """Collate samples, move to device, run model forward pass. Returns (batch, logits, probs)."""
        batch = tre_collate_cached(batch_samples)
        batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        entity_pairs = list(zip(batch['e1_ids'], batch['e2_ids']))
        out = self._model(
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
        logits = out['LF']
        probs = torch.softmax(logits, dim=1)
        return batch, logits, probs

    def _resolve_uid_order(self, chunk, batch_samples, batch):
        """Map batch UIDs back to original PSL data indices."""
        uid_to_orig = {}
        for idx_in_chunk, (orig_idx, _, _) in enumerate(chunk):
            uid_to_orig[batch_samples[idx_in_chunk]['uid']] = orig_idx
        return [uid_to_orig[u] for u in batch['uids']]

    def _evaluate_splits(self):
        """Evaluate Neural Baseline on Valid set during training."""
        split_names = {1: 'VALID'}
        self._model.eval()
        batch_size = 128

        with torch.no_grad():
            for ds_idx in [1]:  # Only Valid
                ds = self._datasets[ds_idx]
                if ds is None or len(ds) == 0:
                    continue

                all_preds, all_labels = [], []
                for start in range(0, len(ds), batch_size):
                    batch_samples = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
                    batch, logits, _ = self._forward_batch(batch_samples)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    for p, l in zip(preds, labels):
                        if int(l) < 5:
                            all_preds.append(int(p))
                            all_labels.append(int(l))

                if len(all_preds) > 0:
                    acc = (np.array(all_preds) == np.array(all_labels)).mean()
                    f1 = f1_score(all_labels, all_preds, average='weighted')
                    is_best = f1 > self._best_valid_f1
                    marker = ' *** BEST' if is_best else ''
                    print(f"  [{split_names[ds_idx]}] Acc= {acc:.6f}, F1-weighted= {f1:.6f}{marker}", flush=True)

                    if is_best and self._save_path:
                        self._best_valid_f1 = f1
                        ckpt_dir = self._save_path
                        os.makedirs(ckpt_dir, exist_ok=True)
                        ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pt')
                        checkpoint = {
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._optimizer.state_dict(),
                            'step': self._step,
                            'fit_call_count': self._fit_call_count,
                            'best_valid_f1': self._best_valid_f1,
                        }
                        torch.save(checkpoint, ckpt_path)
                        print(f"  [SAVE] Best model saved (F1={f1:.6f}) → {ckpt_path}", flush=True)

    # ==================== PSL INTERFACE ====================

    def internal_init_model(self, application, options={}):
        self._application = application

        # Load datasets
        cache_path = options.get('cache-path', '/data/ddao/TRE/pretrained_models/Reasoning/MATRES/cache/')
        if not cache_path.endswith('/'):
            cache_path += '/'

        batch_size = int(options.get('batch-size', 32))
        print(f"Loading datasets from: {cache_path}", flush=True)

        splits = ['train', 'valid', 'test']
        for split in splits:
            loader = create_dataloader(
                cache_path + f"{split}/token_cache_bert.pt",
                cache_path + f"{split}/spacy_cache_bert.jsonl",
                batch_size=batch_size, num_workers=4, pin_memory=True
            )
            self._datasets.append(loader.dataset)

        print(f"Loaded {sum(len(ds) for ds in self._datasets)} samples from {len(splits)} splits.", flush=True)

        # Build ID map: (global_u, global_v) -> (ds_idx, sample_idx)
        # Must replicate the same global ID assignment as generate_psl_data_v2.py
        self._id_map = {}
        global_entity_map = {}
        global_next_id = 0

        for ds_idx, ds in enumerate(self._datasets):
            if ds is None:
                continue
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
                self._id_map[(u, v)] = (ds_idx, i)

        # Init Model
        max_id = max(ds.input_ids.max().item() for ds in self._datasets if ds is not None)
        print(f"Max input_id: {max_id}", flush=True)

        bert = AutoModel.from_pretrained('bert-base-uncased')
        if max_id >= bert.config.vocab_size:
            new_size = max_id + 1
            print(f"Resizing embeddings from {bert.config.vocab_size} to {new_size}")
            bert.resize_token_embeddings(new_size)

        freeze_bert = options.get('freeze-bert', 'false').lower() == 'true'
        print(f"Freeze BERT: {freeze_bert}")

        num_classes = int(options.get('class-size', 3))
        self._model = BaseLine(bert, num_classes=num_classes, freeze_bert=freeze_bert)
        self._model.to(self._device)

        # Load weights if provided
        load_path = options.get('load-path')
        if load_path and os.path.exists(load_path):
            if os.path.isdir(load_path):
                ckpt_path = os.path.join(load_path, "checkpoint.pt")
                legacy_path = os.path.join(load_path, "pytorch_model.bin")
            else:
                ckpt_path = load_path
                legacy_path = load_path

            if os.path.exists(ckpt_path):
                print(f"Loading full checkpoint from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)
                self._model.load_state_dict(ckpt['model_state_dict'])
                # Optimizer will be loaded after creation below
                self._step = ckpt.get('step', 0)
                self._fit_call_count = ckpt.get('fit_call_count', 0)
                _pending_optimizer_state = ckpt.get('optimizer_state_dict')
                print(f"Resumed from step={self._step}, fit_call_count={self._fit_call_count}")
            elif os.path.exists(legacy_path):
                print(f"Loading legacy weights from {legacy_path}")
                state_dict = torch.load(legacy_path, map_location=self._device, weights_only=False)
                try:
                    self._model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: partial load: {e}")
                    model_dict = self._model.state_dict()
                    matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(matched)
                    self._model.load_state_dict(model_dict)
                    print(f"Loaded {len(matched)}/{len(state_dict)} matched layers.")
                _pending_optimizer_state = None
            else:
                _pending_optimizer_state = None
        else:
            _pending_optimizer_state = None

        # Save path for best model
        self._save_path = options.get('save-path')

        # Optimizer
        lr = float(options.get('simple-learning-rate', 1e-4))
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

        # Restore optimizer state if available
        if _pending_optimizer_state is not None:
            try:
                self._optimizer.load_state_dict(_pending_optimizer_state)
                print("Optimizer state restored.")
            except Exception as e:
                print(f"Warning: Could not restore optimizer state: {e}")

        return {}

    def internal_fit(self, data, gradients, options={}):
        """PSL provides gradients w.r.t. probabilities. Apply them to update the neural network."""
        self._model.train()
        data = ensure_native_byteorder(data)
        gradients = ensure_native_byteorder(gradients)
        self._fit_call_count += 1

        print(f"\n>>> [STEP {self._fit_call_count}]", flush=True)

        # Measure total wall-clock (includes PSL inference between steps)
        now = time.time()
        if self._last_fit_end is not None:
            total_wall = now - self._last_fit_end
        else:
            total_wall = None

        batch_size = int(options.get('batch-size', 32))
        samples_to_process = self._map_psl_data(data)

        if len(samples_to_process) == 0:
            return {}

        self._optimizer.zero_grad()
        total_ce, total_ce_count, total_grad = 0.0, 0, 0.0

        for i in range(0, len(samples_to_process), batch_size):
            chunk = samples_to_process[i : i + batch_size]
            batch_samples = [self._datasets[ds_idx][s_idx] for _, ds_idx, s_idx in chunk]

            batch, logits, probs = self._forward_batch(batch_samples)
            sorted_orig_indices = self._resolve_uid_order(chunk, batch_samples, batch)

            # No VAGUE masking needed for TDDMan (all 5 labels valid)

            batch_grads = torch.from_numpy(gradients[sorted_orig_indices]).to(self._device)

            # CE Loss (Neural vs Ground Truth)
            labels = batch['labels']
            valid = labels < 5
            if valid.any():
                ce = torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction='sum')
                total_ce += ce.item()
                total_ce_count += valid.sum().item()

            # Backward with PSL gradients
            probs.backward(gradient=batch_grads)
            total_grad += batch_grads.abs().mean().item()

        ce_avg = total_ce / max(total_ce_count, 1)
        n_batches = len(samples_to_process) / batch_size + 1e-9
        print(f"  [TRAIN] CE Loss: {ce_avg:.6f} | PSL Grad: {total_grad / n_batches:.6f}", flush=True)

        self._optimizer.step()
        self._step += 1

        self._evaluate_splits()
        self._last_fit_end = time.time()
        if total_wall is not None:
            print(f"  [TIME] {self._last_fit_end - now:.0f}s (python) | {total_wall:.0f}s (total)", flush=True)
        else:
            print(f"  [TIME] {self._last_fit_end - now:.0f}s (python)", flush=True)
        print(f"<<< [STEP {self._fit_call_count}]\n", flush=True)

        return {'ce_loss': ce_avg}

    def internal_predict(self, data, options={}):
        """Return softmax probabilities for each (e1, e2) pair sent by PSL."""
        data = ensure_native_byteorder(data)
        self._model.eval()

        if len(data) == 0:
            return np.empty((0, self._model.num_classes), dtype=np.float32), {}

        batch_size = int(options.get('batch-size', 32))
        samples_to_process = self._map_psl_data(data)

        # Default: uniform distribution for unmatched (latent) pairs
        results = np.full((len(data), self._model.num_classes), 1.0 / self._model.num_classes, dtype=np.float32)

        with torch.no_grad():
            for i in range(0, len(samples_to_process), batch_size):
                chunk = samples_to_process[i : i + batch_size]
                batch_samples = [self._datasets[ds_idx][s_idx] for _, ds_idx, s_idx in chunk]

                batch, _, probs = self._forward_batch(batch_samples)
                sorted_orig_indices = self._resolve_uid_order(chunk, batch_samples, batch)

                probs_np = probs.cpu().numpy()
                if np.isnan(probs_np).any():
                    probs_np = np.nan_to_num(probs_np, nan=1.0 / self._model.num_classes)

                results[sorted_orig_indices] = probs_np

        if torch.cuda.is_available():
            print(f"  [PYTORCH_MAX_MEM] {torch.cuda.max_memory_allocated(self._device) / (1024 * 1024):.2f} MB", flush=True)

        return results, {}

    def internal_eval(self, data, options={}):
        return {}

    def internal_save(self, options={}):
        """PSL calls this to save. We save as last_checkpoint.pt to avoid overwriting best model."""
        save_path = options.get('save-path')
        if save_path:
            if not save_path.endswith('.pt') and not save_path.endswith('.bin'):
                os.makedirs(save_path, exist_ok=True)
                ckpt_path = os.path.join(save_path, "last_checkpoint.pt")
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ckpt_path = save_path

            checkpoint = {
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'step': self._step,
                'fit_call_count': self._fit_call_count,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Saved last checkpoint to {ckpt_path} (step={self._step})")
        return {}
