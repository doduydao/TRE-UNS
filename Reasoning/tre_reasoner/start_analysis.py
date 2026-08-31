import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import defaultdict
import os

# =========================================================
# HELPER: DATA NORMALIZATION
# =========================================================
def load_and_normalize(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    # Standardize column names
    # Common variations: 'ground_truth', 'Ground-Truth', 'Ground_Truth_ID', 'prediction', 'Prediction'
    # Target standard: 'doc_id', 'e1_id', 'e2_id', 'ground_truth', 'prediction'
    
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if 'doc' in cl: col_map[c] = 'doc_id'
        elif 'e1' in cl: col_map[c] = 'e1_id'
        elif 'e2' in cl: col_map[c] = 'e2_id'
        elif 'ground' in cl: col_map[c] = 'ground_truth'
        elif 'pred' in cl: col_map[c] = 'prediction'
        
    df = df.rename(columns=col_map)
    
    # Check required columns
    req = ['doc_id', 'e1_id', 'e2_id', 'ground_truth', 'prediction']
    missing = [r for r in req if r not in df.columns]
    if missing:
        raise ValueError(f"File {filepath} missing columns: {missing}. Found: {list(df.columns)}")
        
    return df

# =========================================================
# HELPER: CONSISTENCY CHECK
# =========================================================
def calculate_consistency(df, vague_label='VAGUE'):
    """
    Computes Transitivity and Symmetry violation rates.
    Assumes standard label strings: AFTER, BEFORE, EQUAL, VAGUE/OVERLAP
    Or maps numeric if needed.
    """
    
    # 1. Detect Label Type
    sample_label = df['prediction'].iloc[0]
    is_numeric = isinstance(sample_label, (int, np.integer))
    
    # Map to standard numeric values for logic checks
    # 0: AFTER, 1: BEFORE, 2: EQUAL, 3: VAGUE (MATRES Std)
    # I2B2: 0: BEFORE, 1: AFTER, 2: OVERLAP (Typical) -> Needs mapping
    
    # Naive mapping strategies
    # Strategy: build adjacency using raw labels and map only for transitivity logic
    
    # MATRES Logic
    TRANSITIVITY_MATRES = {
        ('AFTER', 'AFTER'): 'AFTER',
        ('BEFORE', 'BEFORE'): 'BEFORE',
        ('EQUAL', 'EQUAL'): 'EQUAL',
        ('AFTER', 'EQUAL'): 'AFTER',
        ('EQUAL', 'AFTER'): 'AFTER',
        ('BEFORE', 'EQUAL'): 'BEFORE',
        ('EQUAL', 'BEFORE'): 'BEFORE'
    }
    
    SYMMETRY_MATRES = {
        'AFTER': 'BEFORE',
        'BEFORE': 'AFTER',
        'EQUAL': 'EQUAL',
        'VAGUE': 'VAGUE'
    }
    
    # Use generic string logic if the labels are not numeric.
    # If numeric, assume MATRES ID map (0:AFTER, 1:BEFORE, 2:EQUAL).
    
    doc_groups = df.groupby('doc_id')
    total_triplets = 0
    consistent_triplets = 0
    total_pairs = 0
    consistent_pairs = 0
    
    violations = []
    
    for doc_id, group in doc_groups:
        adj = {}
        for _, row in group.iterrows():
            adj[(row['e1_id'], row['e2_id'])] = row['prediction']
            
        # Symmetry
        visited = set()
        for (u, v), r in adj.items():
            if (u,v) in visited: continue
            if (v,u) in adj:
                visited.add((v,u))
                total_pairs += 1
                r_inv = adj[(v,u)]
                
                # Check string symmetry
                expected = SYMMETRY_MATRES.get(str(r).upper())
                if expected and str(r_inv).upper() == expected:
                    consistent_pairs += 1
                elif str(r) == str(r_inv) and str(r).upper() == 'EQUAL': # Equal is sym
                     consistent_pairs += 1
                elif str(r).upper() == 'VAGUE' and str(r_inv).upper() == 'VAGUE':
                     consistent_pairs += 1

        # Transitivity
        # O(N^2) or edges
        nodes = set([x[0] for x in adj.keys()] + [x[1] for x in adj.keys()])
        edges = adj # access by key
        
        # Build neighbor list for speed
        neighbors = defaultdict(list)
        for u, v in adj:
            neighbors[u].append(v)
            
        for u in neighbors:
            for v in neighbors[u]:
                r1 = str(adj[(u,v)]).upper()
                if r1 == str(vague_label).upper(): continue
                
                if v in neighbors:
                    for w in neighbors[v]:
                        if w == u: continue
                        r2 = str(adj.get((v,w))).upper()
                        if r2 == str(vague_label).upper(): continue
                        
                        if w in neighbors[u]: # Check cycle A->B->C and A->C existence
                            r3 = str(adj[(u,w)]).upper()
                            if r3 == str(vague_label).upper(): continue
                            
                            total_triplets += 1
                            
                            # Check Logic
                            expected = TRANSITIVITY_MATRES.get((r1, r2))
                            if expected:
                                if r3 == expected:
                                    consistent_triplets += 1
                                else:
                                    violations.append((doc_id, u, v, w, r1, r2, r3, expected))
                            else:
                                # Undefined transitivity (e.g. BEFORE + AFTER) -> No constraint usually, or VAGUE
                                # We treat as Consistent (Total only counts constrained)
                                # But wait, total_triplets incremented above.
                                total_triplets -= 1 

    tvr = 1.0 - (consistent_triplets / total_triplets) if total_triplets > 0 else 0.0
    svr = 1.0 - (consistent_pairs / total_pairs) if total_pairs > 0 else 0.0
    
    return tvr, svr, total_triplets, violations


# =========================================================
# MAIN LOGIC
# =========================================================
def analyze(args):
    f_out = open(args.output_file, 'w')
    
    def log(s):
        print(s)
        f_out.write(s + "\n")
        
    df1 = load_and_normalize(args.file1)
    df2 = None
    if(args.file2):
        df2 = load_and_normalize(args.file2)
        
    mode = 'compare' if df2 is not None else 'single'
    log(f"Mode: {mode.upper()}\n")
    
    # 1. Single Analysis (Applied to File 1)
    def analyze_single(df, name):
        log(f"--- Analysis: {name} ---")
        # Filter VAGUE/OVERLAP for metrics if requested?
        # Typically we filter label "3" or "VAGUE"
        # Let's detect columns
        labels = df['ground_truth'].unique()
        
        # Determine what is 'VAGUE'
        vague_lbl = 'VAGUE'
        if 3 in labels: vague_lbl = 3
        if 'OVERLAP' in labels: vague_lbl = 'OVERLAP' # I2B2 often excludes OVERLAP in some metrics or keeps it.
        # MATRES usually excludes VAGUE.
        
        # Classification Report
        df_filtered = df[df['ground_truth'] != vague_lbl]
        if len(df_filtered) > 0:
            log(f"Classification Report (Excluding {vague_lbl}):")
            log(classification_report(df_filtered['ground_truth'], df_filtered['prediction'], digits=4))
        else:
            log("Warning: No data left after filtering VAGUE. Showing full report:")
            log(classification_report(df['ground_truth'], df['prediction'], digits=4))
            
        # Consistency
        log("Consistency Analysis (MATRES-style Logic):")
        tvr, svr, n_trip, viols = calculate_consistency(df, vague_label=vague_lbl)
        log(f"Transitivity Violation Rate: {tvr:.4f} ({n_trip} triplets checked)")
        log(f"Symmetry Violation Rate:     {svr:.4f}")
        return tvr, viols
        
    tvr1, viols1 = analyze_single(df1, "File 1")
    
    if mode == 'compare':
        log("\n")
        tvr2, viols2 = analyze_single(df2, "File 2")
        
        log("\n--- Comparison ---")
        # Merge
        merged = pd.merge(df1, df2, on=['doc_id', 'e1_id', 'e2_id'], suffixes=('_1', '_2'))
        
        # Agreement
        agree = (merged['prediction_1'] == merged['prediction_2']).mean()
        log(f"Prediction Agreement: {agree:.4f}")
        
        # Shifts
        shifts = merged[merged['prediction_1'] != merged['prediction_2']]
        log(f"Total Prediction Shifts: {len(shifts)}")
        
        if len(shifts) > 0:
            log("\nTop Shifts (GT | Pred1 -> Pred2):")
            cnt = shifts.groupby(['ground_truth_1', 'prediction_1', 'prediction_2']).size().reset_index(name='count')
            cnt = cnt.sort_values('count', ascending=False).head(20)
            log(cnt.to_string())
            
        # Consistency Improvement
        # Check fixed violations
        # Key: (doc, u, v, w)
        keys1 = set([(v[0], v[1], v[2], v[3]) for v in viols1])
        keys2 = set([(v[0], v[1], v[2], v[3]) for v in viols2])
        
        fixed = keys1 - keys2
        broken = keys2 - keys1
        
        log(f"\nConsistency Changes:")
        log(f"Violations Fixed (in File 1 but not File 2): {len(fixed)}")
        log(f"Violations Broken (in File 2 but not File 1): {len(broken)}")
        
        if len(fixed) > 0 and args.verbose:
            log("\nExample Fixed Violations:")
            count = 0
            for v in viols1:
                k = (v[0], v[1], v[2], v[3])
                if k in fixed:
                    log(str(v))
                    count += 1
                    if count >= 10: break

    f_out.close()
    print(f"Analysis saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Unified Analysis Script")
    parser.add_argument("file1", type=str, help="First CSV prediction file")
    parser.add_argument("file2", type=str, nargs='?', help="Second CSV prediction file (Optional, for comparison)")
    parser.add_argument("--output_file", type=str, default="analysis_report.txt")
    parser.add_argument("--verbose", action="store_true", help="Log detailed violation examples")
    
    args = parser.parse_args()
    analyze(args)

if __name__ == "__main__":
    main()
