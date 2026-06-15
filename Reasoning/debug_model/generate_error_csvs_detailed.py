import pandas as pd
import os
from collections import defaultdict
import argparse

# Global Logic Table
LOGIC_TABLE = {
    ('AFTER', 'AFTER'): 'AFTER', ('BEFORE', 'BEFORE'): 'BEFORE',
    ('EQUAL', 'EQUAL'): 'EQUAL', ('AFTER', 'EQUAL'): 'AFTER',
    ('EQUAL', 'AFTER'): 'AFTER', ('BEFORE', 'EQUAL'): 'BEFORE',
    ('EQUAL', 'BEFORE'): 'BEFORE'
}

def generate_error_csvs(baseline_path, logic_path, out_dir="MATRES/analysis"):
    b_df = pd.read_csv(baseline_path)
    l_df = pd.read_csv(logic_path)

    # Remove VAGUE-labeled samples from comparative error analysis
    b_df = b_df[b_df['ground_truth'] != 'VAGUE'].copy()
    l_df = l_df[l_df['ground_truth'] != 'VAGUE'].copy()
    
    # Merge
    # b_df: doc_id, e1_id, e2_id, ground_truth, prediction (Original Baseline)
    # l_df: doc_id, e1_id, e2_id, ground_truth, encoder_prediction, prediction (Logic Model)
    
    df = pd.merge(b_df, l_df, on=['doc_id', 'e1_id', 'e2_id'], suffixes=('_orig', '_logic_model'))

    encoder_col = None
    for candidate in ['encoder_prediction_logic_model', 'encoder_prediction']:
        if candidate in df.columns:
            encoder_col = candidate
            break
    if encoder_col is None:
        raise ValueError("Cannot find encoder prediction column in merged dataframe")
    
    df = df.rename(columns={
        'ground_truth_orig': 'ground_truth',
        'prediction_orig': 'baseline_pred',
        encoder_col: 'encoder_pred',
        'prediction_logic_model': 'logic_pred'
    })
    
    # Adjacency for proofs
    doc_adj = defaultdict(dict)
    for _, row in l_df.iterrows():
        doc_adj[row['doc_id']][(row['e1_id'], row['e2_id'])] = row['prediction']

    def find_logic_evidence(row, target_col):
        doc_id = row['doc_id']
        u, v = row['e1_id'], row['e2_id']
        target = row[target_col]
        adj = doc_adj[doc_id]
        proofs = []
        for (u1, k), r1 in adj.items():
            if u1 == u and (k, v) in adj:
                r2 = adj[(k, v)]
                if LOGIC_TABLE.get((r1, r2)) == target:
                    proofs.append(f"{u}--({r1})-->{k}--({r2})-->{v}")
        return " | ".join(proofs) if proofs else "N/A"

    # Categorize Repairs (Cases where Baseline was wrong but Logic Model is right)
    repairs = df[(df['baseline_pred'] != df['ground_truth']) & (df['logic_pred'] == df['ground_truth'])].copy()
    
    # Type 1: Neural Gain (Encoder learned to be right)
    # baseline_pred != gt, encoder_pred == gt
    repairs.loc[repairs['encoder_pred'] == repairs['ground_truth'], 'repair_type'] = 'Neural Gain (Training)'
    
    # Type 2: Logical Repair (Inference fixed a still-wrong encoder)
    # baseline_pred != gt, encoder_pred != gt, logic_pred == gt
    repairs.loc[repairs['encoder_pred'] != repairs['ground_truth'], 'repair_type'] = 'Logical Repair (Inference)'
    
    # Add evidence for Logical Repairs
    repairs['logic_evidence'] = repairs.apply(lambda r: find_logic_evidence(r, 'ground_truth') if r['repair_type'] == 'Logical Repair (Inference)' else "N/A (Neural Correct)", axis=1)

    # Categorize Breaks (Cases where Baseline was right but Logic Model is wrong)
    breaks = df[(df['baseline_pred'] == df['ground_truth']) & (df['logic_pred'] != df['ground_truth'])].copy()
    breaks['logic_evidence'] = breaks.apply(lambda r: find_logic_evidence(r, 'logic_pred'), axis=1)

    # Mutual and Persistent
    mutuals = df[(df['baseline_pred'] != df['ground_truth']) & (df['logic_pred'] != df['ground_truth']) & (df['logic_pred'] != df['baseline_pred'])]
    persistent = df[(df['baseline_pred'] != df['ground_truth']) & (df['logic_pred'] == df['baseline_pred'])]

    # Save
    os.makedirs(out_dir, exist_ok=True)
    
    repairs[['doc_id', 'e1_id', 'e2_id', 'ground_truth', 'baseline_pred', 'encoder_pred', 'logic_pred', 'repair_type', 'logic_evidence']].to_csv(f'{out_dir}/logic_repairs_detailed.csv', index=False)
    breaks[['doc_id', 'e1_id', 'e2_id', 'ground_truth', 'baseline_pred', 'encoder_pred', 'logic_pred', 'logic_evidence']].to_csv(f'{out_dir}/logic_breaks_detailed.csv', index=False)
    
    print(f"Deep Analysis Complete.")
    print(f"Total Repairs: {len(repairs)}")
    print(f" - Neural Gains (Training effect): {len(repairs[repairs['repair_type']=='Neural Gain (Training)'])}")
    print(f" - Logical Repairs (Inference effect): {len(repairs[repairs['repair_type']=='Logical Repair (Inference)'])}")
    print(f"Total Breaks: {len(breaks)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detailed error CSVs for baseline vs logic predictions")
    parser.add_argument("--baseline", default="MATRES/prediction_baseline.csv", help="Path to baseline prediction CSV")
    parser.add_argument("--logic", default="MATRES/prediction.csv", help="Path to logic prediction CSV")
    parser.add_argument("--out_dir", default="MATRES/analysis", help="Output directory for analysis CSV files")
    args = parser.parse_args()
    generate_error_csvs(args.baseline, args.logic, args.out_dir)
