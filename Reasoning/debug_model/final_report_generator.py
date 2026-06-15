import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict
import argparse
import os

# Global Logic Table for Transitivity
LOGIC_TABLE = {
    ('AFTER', 'AFTER'): 'AFTER', ('BEFORE', 'BEFORE'): 'BEFORE',
    ('EQUAL', 'EQUAL'): 'EQUAL', ('AFTER', 'EQUAL'): 'AFTER',
    ('EQUAL', 'AFTER'): 'AFTER', ('BEFORE', 'EQUAL'): 'BEFORE',
    ('EQUAL', 'BEFORE'): 'BEFORE'
}

def generate_final_report(baseline_path, logic_path, output_path):
    b_df = pd.read_csv(baseline_path)
    l_df = pd.read_csv(logic_path)

    # Remove VAGUE-labeled samples from comparative error analysis
    b_df = b_df[b_df['ground_truth'] != 'VAGUE'].copy()
    l_df = l_df[l_df['ground_truth'] != 'VAGUE'].copy()
    
    # Standardize and Merge for general metrics
    df = pd.merge(b_df, l_df, on=['doc_id', 'e1_id', 'e2_id'], suffixes=('_b', '_l'))
    df['gt'] = df['ground_truth_b']
    
    agree = (df['prediction_b'] == df['prediction_l']).mean()
    shifts = df[df['prediction_b'] != df['prediction_l']]
    repairs = df[(df['prediction_b'] != df['gt']) & (df['prediction_l'] == df['gt'])]
    breaks = df[(df['prediction_b'] == df['gt']) & (df['prediction_l'] != df['gt'])]
    mutuals = df[(df['prediction_b'] != df['gt']) & (df['prediction_l'] != df['gt']) & (df['prediction_l'] != df['prediction_b'])]
    pure_failures = df[(df['prediction_b'] != df['gt']) & (df['prediction_l'] == df['prediction_b'])]

    # 1. Detect Violations for both
    def get_violations(df_in):
        adj_map = defaultdict(dict)
        for _, row in df_in.iterrows():
            adj_map[row['doc_id']][(row['e1_id'], row['e2_id'])] = row['prediction']
        
        violations = []
        for doc_id, adj in adj_map.items():
            for (u, v), r1 in adj.items():
                if r1 == 'VAGUE': continue
                for (v_match, w), r2 in adj.items():
                    if v_match == v and w != u:
                        r12 = LOGIC_TABLE.get((r1, r2))
                        if r12 and (u, w) in adj:
                            r_actual = adj[(u, w)]
                            if r_actual != 'VAGUE' and r_actual != r12:
                                violations.append((doc_id, u, v, w, r1, r2, r_actual, r12))
        return violations

    viols_b = get_violations(b_df)
    viols_l = get_violations(l_df)

    doc_adj = defaultdict(dict)
    for _, row in l_df.iterrows():
        doc_adj[row['doc_id']][(row['e1_id'], row['e2_id'])] = row['prediction']

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        def log(s): f.write(s + "\n")

        log("# Detailed Comparative Analysis Report (Tabular)\n")
        
        log("## 1. Performance and Violation Metrics")
        log("| Metric | Baseline | Logic-Aware (V2) |")
        log("| :--- | :--- | :--- |")
        log(f"| Accuracy | {df['prediction_b'].eq(df['gt']).mean():.4f} | {df['prediction_l'].eq(df['gt']).mean():.4f} |")
        log(f"| Logic Violation Count (Triplets) | {len(viols_b)} | {len(viols_l)} |")
        log(f"| Violation Reduction Rate | - | {((len(viols_b)-len(viols_l))/len(viols_b)*100 if len(viols_b)>0 else 0):.2f}% |")
        
        log("\n## 2. Label Transition Dynamics")
        log("| Transition Type | Count |")
        log("| :--- | :--- |")
        log(f"| Agreement | {agree*100:.2f}% |")
        log(f"| Total Label Changes | {len(shifts)} |")
        log(f"| Repairs (Logic fixes Baseline) | {len(repairs)} |")
        log(f"| Breaks (Logic harms Baseline) | {len(breaks)} |")

        log("\n## 3. Detailed Repair Cases")
        log("| ID | Document | Pair (u, v) | GT | Baseline | Logic Evidence (u->k->v) |")
        log("| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        repair_idx = 0
        for _, row in repairs.iterrows():
            repair_idx += 1
            u, v = row['e1_id'], row['e2_id']
            adj = doc_adj[row['doc_id']]
            # Find a triplet u->k->v in logic predictions that matches GT
            proofs = []
            for (u1, k), r1 in adj.items():
                if u1 == u and (k, v) in adj:
                    r2 = adj[(k, v)]
                    if LOGIC_TABLE.get((r1, r2)) == row['gt']:
                        proofs.append(f"{u}-({r1})->{k}-({r2})->{v}")
            proof_str = "<br>".join(proofs[:2]) if proofs else "N/A (Complex/Multi-step)"
            log(f"| {repair_idx} | {row['doc_id']} | ({u}, {v}) | {row['gt']} | {row['prediction_b']} | {proof_str} |")

        log("\n## 4. Detailed Break Cases")
        log("| ID | Document | Pair (u, v) | GT | Baseline | Logic | Evidence (Error-inducing chain) |")
        log("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        
        break_idx = 0
        for _, row in breaks.iterrows():
            break_idx += 1
            u, v = row['e1_id'], row['e2_id']
            adj = doc_adj[row['doc_id']]
            # Find a triplet u->k->v in logic predictions that matches Prediction_L
            proofs = []
            for (u1, k), r1 in adj.items():
                if u1 == u and (k, v) in adj:
                    r2 = adj[(k, v)]
                    if LOGIC_TABLE.get((r1, r2)) == row['prediction_l']:
                        proofs.append(f"{u}-({r1})->{k}-({r2})->{v}")
            proof_str = "<br>".join(proofs[:2]) if proofs else "N/A"
            log(f"| {break_idx} | {row['doc_id']} | ({u}, {v}) | {row['gt']} | {row['prediction_b']} | {row['prediction_l']} | {proof_str} |")

        log("\n## 5. Mutual and Persistent Errors")
        log("| Type | Document | Pair (u, v) | GT | Baseline | Logic |")
        log("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in mutuals.iterrows():
            log(f"| Mutual | {row['doc_id']} | ({row['e1_id']}, {row['e2_id']}) | {row['gt']} | {row['prediction_b']} | {row['prediction_l']} |")
        for _, row in pure_failures.iterrows():
            log(f"| Persistent | {row['doc_id']} | ({row['e1_id']}, {row['e2_id']}) | {row['gt']} | {row['prediction_b']} | {row['prediction_l']} |")

        log("\n" + "="*60)
        log("\nEnd of report")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate final comparative report for baseline vs logic predictions")
    parser.add_argument("--baseline", default="MATRES/prediction_baseline.csv", help="Path to baseline prediction CSV")
    parser.add_argument("--logic", default="MATRES/prediction.csv", help="Path to logic prediction CSV")
    parser.add_argument("--out", default="MATRES/analysis/comparison_analysis_v2.txt", help="Output report path")
    args = parser.parse_args()
    generate_final_report(args.baseline, args.logic, args.out)
