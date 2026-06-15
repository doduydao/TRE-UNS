import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Paths
truth_path = "/home/prof/ddao/daodd/phd-dao-do/TRE/notebooks/neupsl-ijcai23/MATRES/data/experiment::matres/split::0/method::neupsl/category-truth-test.txt"
pred_path = "/home/prof/ddao/daodd/phd-dao-do/TRE/notebooks/neupsl-ijcai23/results/MATRES/method::neupsl/experiment::matres/split::0/method::neupsl/inferred-predicates/REL.txt"

# Load truth: u \t v \t label
truth = pd.read_csv(truth_path, sep='\t', names=['u', 'v', 'label'])
truth['key'] = truth['u'].astype(str) + "_" + truth['v'].astype(str)

# Load pred: u \t v \t label \t score (PSL output format)
preds_raw = pd.read_csv(pred_path, sep='\t', names=['u', 'v', 'label', 'score'])
preds_raw['key'] = preds_raw['u'].astype(str) + "_" + preds_raw['v'].astype(str)

# Get max score label for each (u, v)
# REL.txt contains lines like: u \t v \t 0 \t 0.8 / u \t v \t 1 \t 0.2
# We take the label with highest score per (u, v)
best_preds = preds_raw.sort_values(['key', 'score'], ascending=[True, False]).drop_duplicates('key')

# Merge
merged = truth.merge(best_preds[['key', 'label', 'score']], on='key', suffixes=('_true', '_pred'))

y_true = merged['label_true'].astype(int)
y_pred = merged['label_pred'].astype(int)

acc = accuracy_score(y_true, y_pred)
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {acc:.6f}")
print(f"F1 Weighted: {f1_weighted:.6f}")
print(f"Total Samples: {len(merged)}")
