import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast

# Add project and tre_reasoner root paths to allow imports
project_root = str(Path(__file__).resolve().parent.parent)
tre_reasoner_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if tre_reasoner_root not in sys.path:
    sys.path.insert(0, tre_reasoner_root)

from config import get_runtime_configs, CONFIGS
from data import TRECachedDataset, tre_collate_cached
from data_sampler import DocBatchSampler
from model import create_model


class TimingWrapper:
    """Wraps model components to precisely measure CUDA or CPU execution time."""
    def __init__(self, device):
        self.device = device
        self.neural_time = 0.0
        self.reasoning_time = 0.0
        self.orig_neural_forward = None
        self.orig_reasoning_forward = None

    def wrap(self, model):
        # We save original forwards
        self.orig_neural_forward = model.neural.forward
        if hasattr(model, 'reasoning') and model.reasoning is not None:
            self.orig_reasoning_forward = model.reasoning.forward

        # Wrap neural encoder
        def timed_neural_forward(*args, **kwargs):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = self.orig_neural_forward(*args, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            self.neural_time += time.perf_counter() - t0
            return res

        model.neural.forward = timed_neural_forward

        # Wrap reasoning layer if applicable
        if self.orig_reasoning_forward is not None:
            def timed_reasoning_forward(*args, **kwargs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = self.orig_reasoning_forward(*args, **kwargs)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                self.reasoning_time += time.perf_counter() - t0
                return res

            model.reasoning.forward = timed_reasoning_forward

    def unwrap(self, model):
        if self.orig_neural_forward is not None:
            model.neural.forward = self.orig_neural_forward
        if self.orig_reasoning_forward is not None and hasattr(model, 'reasoning') and model.reasoning is not None:
            model.reasoning.forward = self.orig_reasoning_forward

    def reset(self):
        self.neural_time = 0.0
        self.reasoning_time = 0.0


def benchmark_scalability(args):
    # Setup Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"==================================================")
    print(f"SCALABILITY GRID BENCHMARK FOR TRE MODEL ON {args.dataset.upper()}")
    print(f"Split: {args.split} | Device: {device}")
    print(f"==================================================\n")

    # Load configs
    dataset_cfg, model_cfg = get_runtime_configs(args.dataset, conf_path=args.conf)
    
    # Select split
    split = args.split.lower()
    if split == 'train':
        token_path = dataset_cfg.train_token_path
        jsonl_path = dataset_cfg.train_jsonl_path
    elif split == 'val':
        token_path = dataset_cfg.val_token_path
        jsonl_path = dataset_cfg.val_jsonl_path
    elif split == 'test':
        token_path = dataset_cfg.test_token_path
        jsonl_path = dataset_cfg.test_jsonl_path
    else:
        raise ValueError(f"Unknown split: {split}. Use train/val/test.")
    
    if not token_path or not jsonl_path or not os.path.exists(token_path) or not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"{split} split cache not found.\nToken path: {token_path}\nJSONL path: {jsonl_path}"
        )
    
    print(f"Loading {split} dataset from:\n  - Tokens: {token_path}\n  - JSONL: {jsonl_path}")
    full_dataset = TRECachedDataset(token_path, jsonl_path)
    print(f"Loaded {len(full_dataset)} total temporal relation pairs.")
    
    # ----------------------------------------------------
    # Select exactly args.num_pairs (2048) pairs from documents with the highest number of triplets
    # ----------------------------------------------------
    print(f"\nAnalyzing documents to select exactly {args.num_pairs} pairs with the most logic triplets...")
    
    # Group dataset indices and e1, e2 pairs by doc_id
    doc_indices = defaultdict(list)
    doc_pairs = defaultdict(set)
    for idx in range(len(full_dataset)):
        sample = full_dataset[idx]
        d_id = sample['doc_ids']
        e1 = sample['e1_ids']
        e2 = sample['e2_ids']
        doc_indices[d_id].append(idx)
        doc_pairs[d_id].add((e1, e2))

    # Count triplets (u, v, w) where relations exist for (u,v), (v,w), (u,w)
    doc_triplet_counts = {}
    for d_id, pairs in doc_pairs.items():
        adj = defaultdict(set)
        for u, v in pairs:
            adj[u].add(v)
        triplets = 0
        for u in list(adj.keys()):
            for v in adj[u]:
                for w in adj.get(v, set()):
                    if w in adj[u]:
                        triplets += 1
        doc_triplet_counts[d_id] = triplets

    # Sort documents by triplet counts in descending order
    sorted_docs = sorted(doc_triplet_counts.keys(), key=lambda d: doc_triplet_counts[d], reverse=True)

    # Select indices greedily
    selected_indices = []
    selected_docs_breakdown = []
    
    for d_id in sorted_docs:
        pairs_count = len(doc_indices[d_id])
        triplets_count = doc_triplet_counts[d_id]
        
        selected_docs_breakdown.append({
            'doc_id': d_id,
            'pairs': pairs_count,
            'triplets': triplets_count
        })
        
        selected_indices.extend(doc_indices[d_id])
        if len(selected_indices) >= args.num_pairs:
            break

    # Cut off at exactly args.num_pairs
    selected_indices = selected_indices[:args.num_pairs]
    dataset = torch.utils.data.Subset(full_dataset, selected_indices)
    
    print(f"Successfully selected {len(dataset)} pairs across {len(selected_docs_breakdown)} documents.")
    print("Top documents selected:")
    for doc_info in selected_docs_breakdown[:10]:
        print(f"  - Doc ID: {doc_info['doc_id']:<35} | Pairs: {doc_info['pairs']:<3d} | Triplets: {doc_info['triplets']:<3d}")
    if len(selected_docs_breakdown) > 10:
        print(f"  ... and {len(selected_docs_breakdown) - 10} more documents.")
    
    # Load tokenizer and BERT structure
    print("\nLoading BERT encoder...")
    tokenizer = BertTokenizerFast.from_pretrained(model_cfg.bert_model)
    bert = AutoModel.from_pretrained(model_cfg.bert_model)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
    bert.resize_token_embeddings(len(tokenizer))

    # Determine default model path based on dataset
    if args.model_path is None:
        project_root_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = project_root_dir / "artifacts" / "tre_reasoner" / "checkpoints" / f"{args.dataset.upper()}_{args.mode}_model.pt"
        if checkpoint_path.exists():
            args.model_path = str(checkpoint_path)
        else:
            # Fallback: try MATRES checkpoint
            fallback_path = project_root_dir / "artifacts" / "tre_reasoner" / "checkpoints" / "MATRES_baseline_reasoning_model.pt"
            if fallback_path.exists():
                print(f"Warning: No {args.dataset} checkpoint found at {checkpoint_path}. No checkpoint will be loaded.")

    # Initialize model
    print(f"Initializing {args.mode} model...")
    model = create_model(
        mode=args.mode,
        bert=bert,
        num_classes=dataset_cfg.num_classes,
        num_types=1,
        rule_file_path=dataset_cfg.rule_file,
        relation_map=dataset_cfg.relation_map,
        freeze_bert=True,
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
        tokenizer=tokenizer
    )

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
        # Filter state dict to avoid shape mismatch errors
        model_state = model.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    print(f"Warning: Shape mismatch for {k}: checkpoint={v.shape}, model={model_state[k].shape}. Skipping this parameter.")
            else:
                print(f"Warning: Key {k} not found in model state. Skipping.")
        model.load_state_dict(filtered_state, strict=False)
    else:
        print("Warning: No checkpoint loaded. Running benchmark with initialized/default weights. (Forward pass speed remains identical).")

    model.to(device)
    model.eval()

    # Disable early stopping if requested (by setting tolerance to negative value)
    if args.disable_early_stopping:
        print("\n[INFO] Early stopping is DISABLED. Forcing exactly 'max_steps' unrolled iterations.")
        if hasattr(model, 'reasoning') and model.reasoning is not None:
            model.reasoning.tol = -1.0

    # Wrap model components for timing
    timer = TimingWrapper(device)
    timer.wrap(model)

    # Decode lists
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    max_steps_list = [int(steps) for steps in args.max_steps_list.split(",")]

    results = []
    computation_cache = {}

    print("\n" + "=" * 105)
    print(f"{'Batch Size':<12} | {'Max Steps':<10} | {'Total Time':<12} | {'Throughput':<15} | {'Avg Batch Lat':<15} | {'Neural (s)':<10} | {'Reason (s)':<10} | {'Avg Steps':<10} | {'Max GPU Mem':<12}")
    print("=" * 105)

    for bs in batch_sizes:
        # Build Dataloader
        if model_cfg.data_mode == 'doc':
            subset_doc_ids = []
            for i in range(len(dataset)):
                subset_doc_ids.append(dataset[i]['doc_ids'])
            
            sampler = DocBatchSampler(
                subset_doc_ids,
                batch_size=999999,
                shuffle=False,
                max_pairs_per_batch=bs
            )
            dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=tre_collate_cached)
        else:
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=tre_collate_cached)

        # Generate a unique hashable signature for the current dataloader's batches
        if model_cfg.data_mode == 'doc':
            batches_signature = tuple(tuple(b) for b in dataloader.batch_sampler.batches)
        else:
            batches_signature = ("pair", bs)

        for steps in max_steps_list:
            cache_key = (batches_signature, steps)
            if cache_key in computation_cache:
                cached_res = computation_cache[cache_key]
                print(f"{bs:<12d} | {steps:<10d} | {cached_res['total_time']:<10.2f}s | {cached_res['throughput']:<11.1f} prs/s | {cached_res['avg_batch_latency_ms']:<11.1f} ms | {cached_res['neural_time']:<10.2f} | {cached_res['reasoning_time']:<10.2f} | {cached_res['avg_actual_steps']:<10.1f} | {cached_res['max_memory_mb']:<8.1f} MB (cached)")
                results.append({
                    'batch_size': bs,
                    'max_steps': steps,
                    'total_time': cached_res['total_time'],
                    'throughput': cached_res['throughput'],
                    'avg_batch_latency_ms': cached_res['avg_batch_latency_ms'],
                    'neural_time': cached_res['neural_time'],
                    'reasoning_time': cached_res['reasoning_time'],
                    'avg_actual_steps': cached_res['avg_actual_steps'],
                    'max_memory_mb': cached_res['max_memory_mb']
                })
                continue

            if hasattr(model, 'reasoning') and model.reasoning is not None:
                model.reasoning.max_steps = steps

            # Warmup (using a small fraction of dataloader)
            warmup_batches = min(3, len(dataloader))
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    if idx >= warmup_batches:
                        break
                    inputs = {
                        'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device),
                        'word_marks': batch['word_marks'].to(device),
                        'e1_marks': batch['e1_marks'].to(device),
                        'e2_marks': batch['e2_marks'].to(device)
                    }
                    if args.mode != 'baseline':
                        inputs.update({
                            'entity_pairs': list(zip(batch['e1_ids'], batch['e2_ids'])),
                            'doc_ids': batch['doc_ids'],
                            'e1_type_ids': batch['e1_type_ids'].to(device),
                            'e2_type_ids': batch['e2_type_ids'].to(device)
                        })
                    _ = model(**inputs)

            # Reset timers and memory
            timer.reset()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            total_batches = 0
            total_pairs_processed = 0
            actual_steps_accum = 0.0

            # Inference loop
            with torch.no_grad():
                for batch in dataloader:
                    inputs = {
                        'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device),
                        'word_marks': batch['word_marks'].to(device),
                        'e1_marks': batch['e1_marks'].to(device),
                        'e2_marks': batch['e2_marks'].to(device)
                    }
                    if args.mode != 'baseline':
                        inputs.update({
                            'entity_pairs': list(zip(batch['e1_ids'], batch['e2_ids'])),
                            'doc_ids': batch['doc_ids'],
                            'e1_type_ids': batch['e1_type_ids'].to(device),
                            'e2_type_ids': batch['e2_type_ids'].to(device)
                        })
                    
                    out = model(**inputs)
                    total_batches += 1
                    total_pairs_processed += batch['input_ids'].size(0)

                    # Extract actual reasoning steps executed
                    if args.mode != 'baseline' and 'extras' in out:
                        energy_history = out['extras'].get('energy_history', [])
                        if energy_history:
                            actual_steps_accum += energy_history[-1].get('step', 0)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            elapsed_total = end_time - start_time
            throughput = total_pairs_processed / elapsed_total
            avg_batch_latency = (elapsed_total / total_batches) * 1000  # in ms
            avg_actual_steps = actual_steps_accum / total_batches if total_batches > 0 else 0.0
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0.0

            # Print this configuration results immediately
            print(f"{bs:<12d} | {steps:<10d} | {elapsed_total:<10.2f}s | {throughput:<11.1f} prs/s | {avg_batch_latency:<11.1f} ms | {timer.neural_time:<10.2f} | {timer.reasoning_time:<10.2f} | {avg_actual_steps:<10.1f} | {max_mem:<8.1f} MB")
            
            # Save to computation cache
            computation_cache[cache_key] = {
                'total_time': elapsed_total,
                'throughput': throughput,
                'avg_batch_latency_ms': avg_batch_latency,
                'neural_time': timer.neural_time,
                'reasoning_time': timer.reasoning_time,
                'avg_actual_steps': avg_actual_steps,
                'max_memory_mb': max_mem
            }

            results.append({
                'batch_size': bs,
                'max_steps': steps,
                'total_time': elapsed_total,
                'throughput': throughput,
                'avg_batch_latency_ms': avg_batch_latency,
                'neural_time': timer.neural_time,
                'reasoning_time': timer.reasoning_time,
                'avg_actual_steps': avg_actual_steps,
                'max_memory_mb': max_mem
            })

    # Unwrap timers
    timer.unwrap(model)

    # Save results to CSV if requested
    if args.output_file:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f"\nSaved benchmark results to {args.output_file}")

    print("=" * 105)
    print("\nBenchmark successfully completed!")
    print("==================================================")


def main():
    parser = argparse.ArgumentParser(description="Benchmark scalability of TRE model inference")
    parser.add_argument("--dataset", type=str, default="MATRES", choices=list(CONFIGS.keys()))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--conf", type=str, default=None, help="Path to config file")
    parser.add_argument("--mode", type=str, default="baseline_reasoning", choices=["baseline", "baseline_reasoning", "TRER"])
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use, -1 for CPU")
    parser.add_argument("--num_pairs", type=int, default=2048, help="Number of relation pairs to select from highest triplet count documents")
    parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--batch_sizes", type=str, default="32,64,128,256,512,1024", help="Comma-separated batch sizes to benchmark")
    parser.add_argument("--max_steps_list", type=str, default="50,100,500,1000,5000", help="Comma-separated reasoning max steps to benchmark")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Force model to run exactly max_steps instead of stopping when energy converges")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results as CSV")

    args = parser.parse_args()
    benchmark_scalability(args)


if __name__ == "__main__":
    main()
