import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def evaluate_and_compare(neupsl_path, trer_base_path, trer_reasoning_path):
    print("Loading datasets...")
    df_neupsl = pd.read_csv(neupsl_path)
    df_base = pd.read_csv(trer_base_path)
    df_reas = pd.read_csv(trer_reasoning_path)
    
    # Filter out VAGUE
    df_neupsl = df_neupsl[df_neupsl['ground_truth'] != 'VAGUE'].copy()
    df_base = df_base[df_base['ground_truth'] != 'VAGUE'].copy()
    df_reas = df_reas[df_reas['ground_truth'] != 'VAGUE'].copy()
    
    # Align the datasets
    # Merge them to ensure we're comparing the exact same pairs
    merged = df_base[['doc_id', 'e1_id', 'e2_id', 'ground_truth', 'prediction']].rename(
        columns={'prediction': 'pred_trer_base', 'ground_truth': 'gt'}
    )
    
    merged = pd.merge(merged, df_reas[['doc_id', 'e1_id', 'e2_id', 'prediction']], 
                      on=['doc_id', 'e1_id', 'e2_id'], how='inner')
    merged.rename(columns={'prediction': 'pred_trer_reas'}, inplace=True)
    
    merged = pd.merge(merged, df_neupsl[['doc_id', 'e1_id', 'e2_id', 'prediction']], 
                      on=['doc_id', 'e1_id', 'e2_id'], how='inner')
    merged.rename(columns={'prediction': 'pred_neupsl'}, inplace=True)
    
    print(f"\nTotal common valid pairs evaluated (without VAGUE): {len(merged)}")
    
    # Evaluate Accuracy
    acc_base = accuracy_score(merged['gt'], merged['pred_trer_base'])
    acc_reas = accuracy_score(merged['gt'], merged['pred_trer_reas'])
    acc_neupsl = accuracy_score(merged['gt'], merged['pred_neupsl'])
    
    # Evaluate F1-Micro
    f1_base = f1_score(merged['gt'], merged['pred_trer_base'], average='micro')
    f1_reas = f1_score(merged['gt'], merged['pred_trer_reas'], average='micro')
    f1_neupsl = f1_score(merged['gt'], merged['pred_neupsl'], average='micro')
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"TRER Baseline (Shared): Accuracy = {acc_base:.4f}, F1-Micro = {f1_base:.4f}")
    print(f"TRER Reasoning (Ours):  Accuracy = {acc_reas:.4f}, F1-Micro = {f1_reas:.4f}")
    print(f"NeuPSL:                 Accuracy = {acc_neupsl:.4f}, F1-Micro = {f1_neupsl:.4f}")
    
    labels = sorted(merged['gt'].unique())
    print(f"\nLabels order for Confusion Matrix: {labels}")
    
    print("\n--- CONFUSION MATRICES ---")
    print("TRER Baseline Confusion Matrix:")
    print(confusion_matrix(merged['gt'], merged['pred_trer_base'], labels=labels))
    
    print("\nTRER Reasoning Confusion Matrix:")
    print(confusion_matrix(merged['gt'], merged['pred_trer_reas'], labels=labels))

    print("\nNeuPSL Confusion Matrix:")
    print(confusion_matrix(merged['gt'], merged['pred_neupsl'], labels=labels))
    
    # Comparison
    print("\n--- COMPARATIVE ANALYSIS (Relative to Baseline) ---")
    
    # TRER Reasoning Dynamics
    trer_repaired = ((merged['pred_trer_base'] != merged['gt']) & (merged['pred_trer_reas'] == merged['gt'])).sum()
    trer_broken = ((merged['pred_trer_base'] == merged['gt']) & (merged['pred_trer_reas'] != merged['gt'])).sum()
    print("TRER Reasoning vs Baseline:")
    print(f"  Fixed:  {trer_repaired}")
    print(f"  Broken: {trer_broken}")
    print(f"  Net gained: {trer_repaired - trer_broken}")
    
    # NeuPSL Dynamics
    neupsl_repaired = ((merged['pred_trer_base'] != merged['gt']) & (merged['pred_neupsl'] == merged['gt'])).sum()
    neupsl_broken = ((merged['pred_trer_base'] == merged['gt']) & (merged['pred_neupsl'] != merged['gt'])).sum()
    print("\nNeuPSL vs Baseline:")
    print(f"  Fixed:  {neupsl_repaired}")
    print(f"  Broken: {neupsl_broken}")
    print(f"  Net gained: {neupsl_repaired - neupsl_broken}")

    print("\n--- DIRECT COMPARISON (TRER Reasoning vs NeuPSL) ---")
    trer_better = ((merged['pred_neupsl'] != merged['gt']) & (merged['pred_trer_reas'] == merged['gt'])).sum()
    neupsl_better = ((merged['pred_neupsl'] == merged['gt']) & (merged['pred_trer_reas'] != merged['gt'])).sum()
    both_correct = ((merged['pred_neupsl'] == merged['gt']) & (merged['pred_trer_reas'] == merged['gt'])).sum()
    neither_correct = ((merged['pred_neupsl'] != merged['gt']) & (merged['pred_trer_reas'] != merged['gt'])).sum()
    
    print(f"TRER Reasoning is correct but NeuPSL is wrong: {trer_better}")
    print(f"NeuPSL is correct but TRER Reasoning is wrong: {neupsl_better}")
    print(f"Both models are correct:                       {both_correct}")
    print(f"Neither model is correct:                      {neither_correct}")
    
    
if __name__ == '__main__':
    print("========================================")
    print("          MATRES EVALUATION             ")
    print("========================================")
    evaluate_and_compare(
        'all_results/MATRES/NeuPSL_prediction.csv',
        'all_results/MATRES/TRER_prediction_baseline.csv',
        'all_results/MATRES/TRER_prediction_reasoning.csv'
    )
    
    print("\n\n========================================")
    print("            TBD EVALUATION              ")
    print("========================================")
    try:
        evaluate_and_compare(
            'all_results/TBD/NeuPSL_prediction.csv',
            'all_results/TBD/TRER_prediction_baseline.csv',
            'all_results/TBD/TRER_prediction_reasoning.csv'
        )
    except FileNotFoundError:
        print("TBD results not found yet.")
