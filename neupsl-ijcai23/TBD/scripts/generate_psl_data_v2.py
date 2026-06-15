import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tre.data import create_dataloader

def generate_v2_psl_data(cache_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    
    # Dictionaries to maintain our Global ID system
    # Key: (doc_id, e1_id) -> Value: Global_Integer_ID
    global_entity_map = {}
    next_global_id = 0
    
    # We also keep track of known ground-truth edges to avoid duplicating lines
    # key: split, value: set of (u, v)
    truth_uv_written = {s: set() for s in splits}
    target_lines_written = {s: set() for s in splits}
    
    # Accumulate all entities to write to entity_id_map.txt at the end
    all_unique_entities = set()
    
    # TBD label mapping (LabelEncoder alphabetical):
    # AFTER=0, BEFORE=1, INCLUDES=2, IS_INCLUDED=3, SIMULTANEOUS=4, VAGUE=5
    # VAGUE (label 5) is excluded from training and evaluation
    NUM_CLASSES = 5  # Non-VAGUE classes (0-4)
    VAGUE_LABEL = 5
    
    for split in splits:
        print(f"Processing split: {split}")
        token_path = os.path.join(cache_dir, split, "token_cache_bert.pt")
        spacy_path = os.path.join(cache_dir, split, "spacy_cache_bert.jsonl")
        
        if not os.path.exists(token_path) or not os.path.exists(spacy_path):
            print(f"  -> Missing cache files for {split}, skipping.")
            continue
            
        # We process sequentially, batch size 1 is fine for data extraction
        loader = create_dataloader(token_path, spacy_path, batch_size=1, shuffle=False)
        
        truth_file = os.path.join(output_dir, f"category-truth-{split}.txt")
        target_file = os.path.join(output_dir, f"category-target-{split}.txt")
        
        # We will collect all pairs in this split for Transitive Closure computation
        # doc_id -> list of (u, v, lbl)
        doc_edges = {}
        
        # 1. READ DATALOADER AND ASSIGN GLOBAL IDs
        with open(truth_file, 'w') as f_truth, open(target_file, 'w') as f_target:
            count = 0
            vague_count = 0
            for batch in loader:
                # Dataloader with batch_size=1 returns lists/tensors of size 1
                doc_id = batch['doc_ids'][0]
                e1_raw = batch['e1_ids'][0]  # This could be tensor or int, convert to string
                e2_raw = batch['e2_ids'][0]
                
                if hasattr(e1_raw, 'item'): e1_raw = e1_raw.item()
                if hasattr(e2_raw, 'item'): e2_raw = e2_raw.item()
                
                e1_str = str(e1_raw)
                e2_str = str(e2_raw)
                
                # Fetch True Label
                lbl = int(batch['labels'][0].item())
                
                # Assign Global IDs
                # Assign Global IDs strictly disjoint across splits
                k1 = (split, doc_id, e1_str)
                k2 = (split, doc_id, e2_str)
                
                if k1 not in global_entity_map:
                    global_entity_map[k1] = next_global_id
                    next_global_id += 1
                if k2 not in global_entity_map:
                    global_entity_map[k2] = next_global_id
                    next_global_id += 1
                    
                u = global_entity_map[k1]
                v = global_entity_map[k2]
                
                all_unique_entities.add(u)
                all_unique_entities.add(v)
                
                # Store for Transitive Closure
                if doc_id not in doc_edges:
                    doc_edges[doc_id] = []
                doc_edges[doc_id].append((u, v, lbl))
                
                # ---- A. WRITE TRUTH FILE ----
                # TBD: Skip VAGUE (label 5) from truth
                if lbl == VAGUE_LABEL:
                    vague_count += 1
                elif (u, v) not in truth_uv_written[split]:
                    t_line = f"{u}\t{v}\t{lbl}\n"
                    f_truth.write(t_line)
                    truth_uv_written[split].add((u, v))
                
                # ---- B. WRITE TARGETS ----
                # Targets must contain ALL pairs, including truth pairs.
                # During Learning, if truth pairs are NOT in targets, they are NEVER predicted 
                # and Neural Network will NOT receive gradients for supervised labels!
                for class_lbl in range(NUM_CLASSES):
                    tar_line = f"{u}\t{v}\t{class_lbl}\n"
                    if tar_line not in target_lines_written[split]:
                        f_target.write(tar_line)
                        target_lines_written[split].add(tar_line)
                count += 1
            
            print(f"  -> Read {count} batches ({vague_count} VAGUE pairs excluded from truth).")
            
            # 2. TRANSITIVE CLOSURE ON TARGETS
            # We want PSL to infer relationships between A and C if we know A->B and B->C
            # even if (A,C) is not evaluated by PyTorch metrics.
            latent_count = 0
            for d, edges in doc_edges.items():
                # Build adjacency
                adj = {}
                for u, v, _ in edges:
                    if u not in adj: adj[u] = set()
                    adj[u].add(v)
                    # For candidate generation, assume undirected connection means they co-occur.
                    if v not in adj: adj[v] = set()
                    adj[v].add(u)
                    
                # Find connected components or just 2-hop reachability
                nodes = list(adj.keys())
                for u_node in list(adj.keys()):
                    for v_node in list(adj[u_node]):
                        for w_node in list(adj[v_node]):
                            if w_node != u_node and w_node not in adj[u_node]:
                                # Found a latent pair (u_node, w_node)
                                # Write to target file for PSL to infer
                                for class_lbl in range(NUM_CLASSES):
                                    tar_line = f"{u_node}\t{w_node}\t{class_lbl}\n"
                                    if tar_line not in target_lines_written[split]:
                                        f_target.write(tar_line)
                                        target_lines_written[split].add(tar_line)
                                        
                                        tar_line_sym = f"{w_node}\t{u_node}\t{class_lbl}\n"
                                        if tar_line_sym not in target_lines_written[split]:
                                            f_target.write(tar_line_sym)
                                            target_lines_written[split].add(tar_line_sym)
                                        latent_count += 1
                                        
            print(f"  -> Generated {latent_count} latent target lines.")

    # 3. WRITE ENTITY MAP FILE
    map_file = os.path.join(output_dir, "entity-data-map.txt")
    print(f"\nWriting Relation Entity Map to {map_file}...")
    
    # To be safe, we collect all (u, v) pairs that were actually written to targets
    all_pairs = []
    for s in splits:
        for line in target_lines_written[s]:
             # line is "u\tv\tlabel\n"
             parts = line.strip().split('\t')
             u, v = int(parts[0]), int(parts[1])
             all_pairs.append((u, v))
    
    # Unique pairs in stable order
    unique_pairs = sorted(list(set(all_pairs)))
    
    with open(map_file, 'w') as f:
        for u, v in unique_pairs:
            # First 2 columns (u, v) are consumed by Java (entity-argument-indexes: "0,1")
            # Last 2 columns (u, v) are sent to Python as 'data'
            f.write(f"{u}\t{v}\t{u}\t{v}\n")
    print(f"  -> Wrote {len(unique_pairs)} pair mappings.")

if __name__ == "__main__":
    import argparse
    # Default output_dir: TBD/data/ (relative to this script's location)
    _THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_OUTPUT = os.path.join(_THIS_SCRIPT_DIR, '..', 'data')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/data/ddao/TRE/pretrained_models/Reasoning/TBD/cache")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    
    print("=== PIPELINE V2 TBD DATA GENERATION ===")
    generate_v2_psl_data(args.cache_dir, args.output_dir)
    print("Done!")
