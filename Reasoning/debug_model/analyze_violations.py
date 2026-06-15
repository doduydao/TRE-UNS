import pandas as pd
from collections import defaultdict
import argparse
import os

# Global Logic Table
LOGIC_TABLE = {
    ('AFTER', 'AFTER'): 'AFTER', ('BEFORE', 'BEFORE'): 'BEFORE',
    ('EQUAL', 'EQUAL'): 'EQUAL', ('AFTER', 'EQUAL'): 'AFTER',
    ('EQUAL', 'AFTER'): 'AFTER', ('BEFORE', 'EQUAL'): 'BEFORE',
    ('EQUAL', 'BEFORE'): 'BEFORE'
}

def analyze_remaining_violations(logic_results_path, output_path):
    df = pd.read_csv(logic_results_path)
    df = df[df['ground_truth'] != 'VAGUE'].copy()
    
    # Build adjacency
    doc_adj = defaultdict(dict)
    for _, row in df.iterrows():
        doc_adj[row['doc_id']][(row['e1_id'], row['e2_id'])] = row['prediction']
        
    violation_records = []
    
    for doc_id, adj in doc_adj.items():
        # Check Transitivity: u->v, v->w => u->w
        for (u, v), r1 in adj.items():
            if r1 == 'VAGUE': continue
            for (v_match, w), r2 in adj.items():
                if v_match == v and w != u:
                    expected = LOGIC_TABLE.get((r1, r2))
                    if expected and (u, w) in adj:
                        actual = adj[(u, w)]
                        if actual != 'VAGUE' and actual != expected:
                            violation_records.append({
                                'doc_id': doc_id,
                                'u': u, 'v': v, 'w': w,
                                'rel_uv': r1,
                                'rel_vw': r2,
                                'rel_uw_actual': actual,
                                'rel_uw_expected': expected
                            })
                            
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    viol_df = pd.DataFrame(violation_records)
    viol_df.to_csv(output_path, index=False)
    print(f"Found {len(viol_df)} remaining violations.")
    print(f"Violations saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze remaining transitivity violations")
    parser.add_argument("--pred", default="MATRES/prediction.csv", help="Path to prediction CSV")
    parser.add_argument("--out", default="MATRES/analysis/remaining_violations.csv", help="Output CSV path")
    args = parser.parse_args()
    analyze_remaining_violations(args.pred, args.out)
