#!/usr/bin/env python3
import os
import sys
import json
import time
import re
import math
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import TRECachedDataset from TBD (which symlinks to MATRES/tre)
try:
    sys.path.insert(0, str(PROJECT_ROOT / "TBD"))
    from tre.data import TRECachedDataset
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "MATRES"))
    from tre.data import TRECachedDataset

# TBD has 5 classes: AFTER=0, BEFORE=1, INCLUDES=2, IS_INCLUDED=3, SIMULTANEOUS=4
# VAGUE (label 5) is excluded
NUM_CLASSES = 5
VAGUE_LABEL = 5

def get_free_port():
    """Find a dynamically free port using socket."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def select_and_generate_benchmark_data(cache_dir, output_dir, num_pairs=2048):
    """Select pairs from test set with the most triplets and write benchmark files.
    
    Uses the TEST split of TimeBank-Dense (TBD) dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    datasets = []
    
    print("Loading datasets to construct global ID map...")
    for split in splits:
        token_path = os.path.join(cache_dir, split, "token_cache_bert.pt")
        spacy_path = os.path.join(cache_dir, split, "spacy_cache_bert.jsonl")
        
        if not os.path.exists(token_path) or not os.path.exists(spacy_path):
            raise FileNotFoundError(f"Missing cache files for {split} split at {cache_dir}")
            
        ds = TRECachedDataset(token_path, spacy_path)
        datasets.append(ds)
        
    print(f"Loaded splits: train={len(datasets[0])}, valid={len(datasets[1])}, test={len(datasets[2])}")
    
    # Replicate global ID assignment from generate_psl_data_v2.py
    global_entity_map = {}
    next_global_id = 0
    
    for ds_idx, ds in enumerate(datasets):
        split = splits[ds_idx]
        for i in range(len(ds.e1_ids)):
            e1 = ds.e1_ids[i].item() if hasattr(ds.e1_ids[i], 'item') else ds.e1_ids[i]
            e2 = ds.e2_ids[i].item() if hasattr(ds.e2_ids[i], 'item') else ds.e2_ids[i]
            doc = ds.doc_ids[i]

            for entity in [e1, e2]:
                k = (split, str(doc), str(entity))
                if k not in global_entity_map:
                    global_entity_map[k] = next_global_id
                    next_global_id += 1
                    
    # Select from TEST split (index 2)
    test_ds = datasets[2]
    doc_indices = defaultdict(list)
    doc_pairs = defaultdict(set)
    
    for i in range(len(test_ds)):
        d_id = test_ds.doc_ids[i]
        e1 = test_ds.e1_ids[i].item() if hasattr(test_ds.e1_ids[i], 'item') else test_ds.e1_ids[i]
        e2 = test_ds.e2_ids[i].item() if hasattr(test_ds.e2_ids[i], 'item') else test_ds.e2_ids[i]
        doc_indices[d_id].append(i)
        doc_pairs[d_id].add((e1, e2))

    # Count triplets (u, v, w) in adjacency list
    doc_triplet_counts = {}
    for d_id, pairs in doc_pairs.items():
        adj = defaultdict(set)
        for u_e, v_e in pairs:
            adj[u_e].add(v_e)
        triplets = 0
        for u_e in list(adj.keys()):
            for v_e in adj[u_e]:
                for w_e in adj.get(v_e, set()):
                    if w_e in adj[u_e]:
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
        if len(selected_indices) >= num_pairs:
            break

    selected_indices = selected_indices[:num_pairs]
    print(f"Selected {len(selected_indices)} test pairs across {len(selected_docs_breakdown)} documents.")
    
    # Write targets, truth, and entity map for benchmark
    target_file = os.path.join(output_dir, "category-target-benchmark.txt")
    truth_file = os.path.join(output_dir, "category-truth-benchmark.txt")
    map_file = os.path.join(output_dir, "entity-data-map-benchmark.txt")
    
    with open(target_file, "w") as f_tar, open(truth_file, "w") as f_tru, open(map_file, "w") as f_map:
        for idx in selected_indices:
            e1 = test_ds.e1_ids[idx].item() if hasattr(test_ds.e1_ids[idx], 'item') else test_ds.e1_ids[idx]
            e2 = test_ds.e2_ids[idx].item() if hasattr(test_ds.e2_ids[idx], 'item') else test_ds.e2_ids[idx]
            doc = test_ds.doc_ids[idx]
            lbl = int(test_ds.labels[idx])
            
            u = global_entity_map[('test', str(doc), str(e1))]
            v = global_entity_map[('test', str(doc), str(e2))]
            
            # Targets for all 5 classes (0, 1, 2, 3, 4)
            for class_lbl in range(NUM_CLASSES):
                f_tar.write(f"{u}\t{v}\t{class_lbl}\n")
                
            # Truth (skip VAGUE label 5)
            if lbl < VAGUE_LABEL:
                f_tru.write(f"{u}\t{v}\t{lbl}\n")
                
            # Entity data map line
            f_map.write(f"{u}\t{v}\t{u}\t{v}\n")
            
    print(f"Successfully generated benchmark data files in {output_dir}:")
    print(f"  - Targets: {target_file} ({NUM_CLASSES} classes per pair)")
    print(f"  - Truth:   {truth_file}")
    print(f"  - Map:     {map_file}")

def run_neupsl_infer(cli_dir, config_path, temp_config_path, bs, steps, port, output_dir):
    """Modify configuration to point to benchmark targets, execute NeuPSL, and parse log metrics."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 1. Update options
    if "options" not in config:
        config["options"] = {}
    config["options"]["admmreasoner.maxiterations"] = steps
    config["options"]["predicate.deep.python.port"] = port
    
    # 2. Update Neural/3 predicate options and target file paths
    if "predicates" in config:
        if "Neural/3" in config["predicates"]:
            pred_info = config["predicates"]["Neural/3"]
            if "options" not in pred_info:
                pred_info["options"] = {}
            pred_info["options"]["batch-size"] = str(bs)
            pred_info["options"]["predicate.deep.python.port"] = port
            pred_info["options"]["entity-data-map-path"] = "../data/entity-data-map-benchmark.txt"
            
            # Point to benchmark targets
            pred_info["targets"] = {
                "infer": [ "../data/category-target-benchmark.txt" ]
            }
            
        # 3. Update Rel/3 targets and truth
        if "Rel/3" in config["predicates"]:
            rel_info = config["predicates"]["Rel/3"]
            rel_info["targets"] = {
                "infer": [ "../data/category-target-benchmark.txt" ]
            }
            rel_info["truth"] = {
                "infer": [ "../data/category-truth-benchmark.txt" ]
            }
                
    # Save modified config to temp path
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    # Prepare execution command
    cmd = [
        "java", "-jar", "psl-cli-2.4.0.jar",
        "--infer",
        "--config", os.path.basename(temp_config_path),
        "--output", os.path.basename(output_dir)
    ]
    
    print(f"  [RUN] Command: {' '.join(cmd)}")
    
    # Export NEUPSL_LOG_DIR to the output directory so Python logs go there
    env = os.environ.copy()
    os.makedirs(output_dir, exist_ok=True)
    env["NEUPSL_LOG_DIR"] = str(output_dir)
    
    start_time = time.perf_counter()
    
    process = subprocess.Popen(
        cmd,
        cwd=str(cli_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    stdout_lines = []
    
    # Regex patterns
    re_predict_start = re.compile(r"^(\d+)\s+\[main\]\s+DEBUG\s+.*Predict deep model\s+org")
    re_predict_end = re.compile(r"^(\d+)\s+\[main\]\s+DEBUG\s+.*Predict deep model result")
    re_opt_metrics = re.compile(r"Total Optimization Time:\s*(\d+),\s*Total Number of Iterations:\s*(\d+)")
    
    predict_start_ms = None
    predict_end_ms = None
    reasoning_time_ms = None
    actual_iterations = None
    
    # Process stdout line by line
    while True:
        line = process.stdout.readline()
        if not line:
            break
        stdout_lines.append(line)
        
        match_start = re_predict_start.search(line)
        if match_start:
            predict_start_ms = int(match_start.group(1))
            
        match_end = re_predict_end.search(line)
        if match_end:
            predict_end_ms = int(match_end.group(1))
            
        match_opt = re_opt_metrics.search(line)
        if match_opt:
            reasoning_time_ms = int(match_opt.group(1))
            actual_iterations = int(match_opt.group(2))
            
    stderr_content = process.stderr.read()
    process.wait()
    
    end_time = time.perf_counter()
    elapsed_total = end_time - start_time
    
    neural_time = None
    if predict_start_ms is not None and predict_end_ms is not None:
        neural_time = (predict_end_ms - predict_start_ms) / 1000.0
        
    reasoning_time = None
    if reasoning_time_ms is not None:
        reasoning_time = reasoning_time_ms / 1000.0
        
    # Read PyTorch GPU Max Memory from neupsl_python.log
    max_memory_mb = 0.0
    log_file_path = Path(output_dir) / "neupsl_python.log"
    if log_file_path.exists():
        with open(log_file_path, "r") as lf:
            for log_line in lf:
                if "[PYTORCH_MAX_MEM]" in log_line:
                    match_mem = re.search(r"\[PYTORCH_MAX_MEM\]\s*([\d\.]+)\s*MB", log_line)
                    if match_mem:
                        max_memory_mb = float(match_mem.group(1))
                        
    return {
        "status": process.returncode,
        "total_time": elapsed_total,
        "neural_time": neural_time,
        "reasoning_time": reasoning_time,
        "actual_iterations": actual_iterations,
        "max_memory_mb": max_memory_mb,
        "error": stderr_content if process.returncode != 0 else None
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark scalability of NeuPSL inference on TimeBank-Dense (TBD) test set")
    parser.add_argument("--batch_sizes", type=str, default="32,64,128,256,512,1024", help="Comma-separated batch sizes to benchmark")
    parser.add_argument("--max_steps_list", type=str, default="50,100,500,1000,5000", help="Comma-separated reasoning max iterations to benchmark")
    parser.add_argument("--num_pairs", type=int, default=2048, help="Number of relation pairs to select from highest triplet count documents")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results as CSV")
    args = parser.parse_args()
    
    # Cache and CLI directories setup for TBD
    cache_dir = "/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache"
    cli_dir = PROJECT_ROOT / "TBD" / "cli"
    data_dir = PROJECT_ROOT / "TBD" / "data"
    base_config = cli_dir / "TBD-infer.json"
    
    if not base_config.exists():
        print(f"Base config not found: {base_config}")
        sys.exit(1)
        
    # Generate benchmark data files from test set
    print("==================================================")
    print("GENERATING BENCHMARK DATASET SUBSET (TBD TEST)")
    select_and_generate_benchmark_data(cache_dir, data_dir, num_pairs=args.num_pairs)
    print("==================================================\n")
    
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    max_steps_list = [int(x.strip()) for x in args.max_steps_list.split(",") if x.strip()]
    
    print("==================================================")
    print("SCALABILITY GRID BENCHMARK FOR NeuPSL ON TBD")
    print(f"Base Config: {base_config}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Max Steps:   {max_steps_list}")
    print("==================================================\n")
    
    results = []
    
    print("=" * 125)
    print(f"{'Batch Size':<12} | {'Max Steps':<10} | {'Total Time':<12} | {'Throughput':<15} | {'Avg Batch Lat':<15} | {'Neural (s)':<10} | {'Reason (s)':<10} | {'Avg Steps':<10} | {'Max GPU Mem':<12}")
    print("=" * 125)
    
    # Run the grid
    for bs in batch_sizes:
        for steps in max_steps_list:
            port = get_free_port()
            temp_config = cli_dir / f"TBD-infer-temp-{bs}-{steps}.json"
            temp_output_dir = cli_dir / f"inferred-predicates-temp-{bs}-{steps}"
            
            if temp_output_dir.exists():
                import shutil
                shutil.rmtree(temp_output_dir)
                
            try:
                run_res = run_neupsl_infer(
                    cli_dir=cli_dir,
                    config_path=base_config,
                    temp_config_path=temp_config,
                    bs=bs,
                    steps=steps,
                    port=port,
                    output_dir=temp_output_dir
                )
                
                if run_res["status"] == 0:
                    total_time = run_res["total_time"]
                    throughput = args.num_pairs / total_time
                    
                    # Compute total batches (targets = num_pairs * NUM_CLASSES)
                    total_batches = math.ceil(args.num_pairs * NUM_CLASSES / bs)
                    avg_batch_latency = (total_time / total_batches) * 1000
                    
                    neural_str = f"{run_res['neural_time']:.2f}" if run_res['neural_time'] is not None else "N/A"
                    reason_str = f"{run_res['reasoning_time']:.2f}" if run_res['reasoning_time'] is not None else "N/A"
                    steps_str = f"{run_res['actual_iterations']:.1f}" if run_res['actual_iterations'] is not None else "N/A"
                    mem_str = f"{run_res['max_memory_mb']:.1f} MB"
                    
                    print(f"{bs:<12d} | {steps:<10d} | {total_time:<10.2f}s | {throughput:<11.1f} prs/s | {avg_batch_latency:<11.1f} ms | {neural_str:<10} | {reason_str:<10} | {steps_str:<10} | {mem_str:<12}")
                    
                    results.append({
                        'batch_size': bs,
                        'max_steps': steps,
                        'total_time': total_time,
                        'throughput': throughput,
                        'avg_batch_latency_ms': avg_batch_latency,
                        'neural_time': run_res['neural_time'],
                        'reasoning_time': run_res['reasoning_time'],
                        'avg_actual_steps': run_res['actual_iterations'],
                        'max_memory_mb': run_res['max_memory_mb']
                    })
                else:
                    print(f"{bs:<12d} | {steps:<10d} | FAILED (Status {run_res['status']})")
                    print(f"Error log:\n{run_res['error']}")
            except Exception as e:
                print(f"{bs:<12d} | {steps:<10d} | EXCEPTION: {e}")
            finally:
                if temp_config.exists():
                    os.remove(temp_config)
                if temp_output_dir.exists():
                    import shutil
                    shutil.rmtree(temp_output_dir)
            
            # Delay to let ports release properly
            time.sleep(2)
            
    print("=" * 125)
    print("\nBenchmark completed successfully!")
    
    # Save CSV
    if args.output_file and results:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Saved results to {output_path}")
        except ImportError:
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved results (via csv module) to {output_path}")

if __name__ == "__main__":
    main()
