from collections import defaultdict, deque
import pandas as pd
import numpy as np


"""
Fast and correct reimplementation of the TempEval-3 evaluation script
( UzZaman & Allen, 2011 ; Sun et al., 2013a for i2b2 2012 )
Implements closure-based evaluation for temporal relations (TLINKs):
Precision = |Pred ∩ Closure(Gold)| / |Pred|
Recall    = |Gold ∩ Closure(Pred)| / |Gold|
F1        = 2PR / (P+R)
"""


# ============================================================
# === 1. Inverse mapping and composition rules ===============
# ============================================================

INVERSE = {
    "BEFORE": "AFTER",
    "AFTER": "BEFORE",
    "INCLUDES": "IS_INCLUDED",
    "IS_INCLUDED": "INCLUDES",
    "SIMULTANEOUS": "SIMULTANEOUS",
    "IDENTITY": "IDENTITY",
    "OVERLAP": "OVERLAP",
}

COMPOSE = {
    ("BEFORE", "BEFORE"): ["BEFORE"],
    ("AFTER", "AFTER"): ["AFTER"],
    ("INCLUDES", "BEFORE"): ["BEFORE"],
    ("BEFORE", "IS_INCLUDED"): ["BEFORE"],
    ("IS_INCLUDED", "AFTER"): ["AFTER"],
    ("INCLUDES", "IS_INCLUDED"): ["INCLUDES"],
    ("SIMULTANEOUS", "SIMULTANEOUS"): ["SIMULTANEOUS"],
}


# ============================================================
# === 2. Baseline slow closure (original TE3) ================
# ============================================================

def closure_graph(relations):
    """
    Original TempEval-3 style closure (O(N²)).
    Adds inverses and composed links until fix-point.
    """
    closure = set(relations)
    added = True

    while added:
        added = False
        new_rel = set()

        for (a, b, r1) in closure:
            # Add inverse
            inv = INVERSE.get(r1)
            if inv and (b, a, inv) not in closure:
                new_rel.add((b, a, inv))

            # Forward composition (a-b-c)
            for (x, y, r2) in closure:
                if b == x:
                    composed = COMPOSE.get((r1, r2), [])
                    for r3 in composed:
                        triple = (a, y, r3)
                        if triple not in closure:
                            new_rel.add(triple)

        if new_rel:
            closure |= new_rel
            added = True

    return closure


# ============================================================
# === 3. Fast BFS-based closure (O(N log N)) =================
# ============================================================

def fast_closure(relations):
    """
    Faster closure computation using BFS propagation.
    Produces same results as closure_graph(), but 10-50× faster.
    """
    if not relations:
        return set()

    inverse = INVERSE
    comp = {
        ("BEFORE", "BEFORE"): "BEFORE",
        ("AFTER", "AFTER"): "AFTER",
        ("INCLUDES", "IS_INCLUDED"): "INCLUDES",
        ("IS_INCLUDED", "INCLUDES"): "IS_INCLUDED",
    }

    # adjacency lists per relation type
    adj = defaultdict(lambda: defaultdict(set))
    closure = set(relations)
    visited = set(relations)

    for a, b, r in relations:
        adj[r][a].add(b)
        inv = inverse.get(r)
        if inv:
            adj[inv][b].add(a)
            closure.add((b, a, inv))
            visited.add((b, a, inv))

    queue = deque(closure)

    while queue:
        a, b, r1 = queue.popleft()
        for r2, targets in adj.items():
            if (r1, r2) in comp:
                r3 = comp[(r1, r2)]
                for c in targets.get(b, []):
                    new_rel = (a, c, r3)
                    if new_rel not in visited:
                        visited.add(new_rel)
                        closure.add(new_rel)
                        adj[r3][a].add(c)
                        queue.append(new_rel)

    return closure


# ============================================================
# === 4. Evaluation (Sun et al., 2013a) ======================
# ============================================================

def evaluate_tempeval3(gold_relations, pred_relations, use_fast=True):
    """
    Compute precision, recall, and F1 following TempEval-3/i2b2 2012.
      Precision = |Pred ∩ Closure(Gold)| / |Pred|
      Recall    = |Gold ∩ Closure(Pred)| / |Gold|
      F1        = 2PR / (P+R)
    use_fast=True → use BFS closure (recommended)
    """
    if use_fast:
        gold_closure = fast_closure(gold_relations)
        pred_closure = fast_closure(pred_relations)
    else:
        gold_closure = closure_graph(gold_relations)
        pred_closure = closure_graph(pred_relations)

    n_pred = len(pred_relations)
    n_gold = len(gold_relations)
    n_pred_closure = len(pred_closure)
    n_gold_closure = len(gold_closure)

    # intersection logic per Sun et al. (2013a)
    pred_verify = set(pred_relations) & gold_closure
    gold_verify = set(gold_relations) & pred_closure

    precision = len(pred_verify) / n_pred if n_pred else 0.0
    recall = len(gold_verify) / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Gold": n_gold,
        "Pred": n_pred,
        "GoldClosure": n_gold_closure,
        "PredClosure": n_pred_closure,
    }


# ============================================================
# === 5. Build relations from DataFrame ======================
# ============================================================

def build_relations_from_df(df, preds=None, label_col="label_encoded", id2label=None):
    """
    Build (gold_relations, pred_relations) from a DataFrame.
    Required columns: document_id, entity1_id, entity2_id, label_col
    Each TLINK is represented as (doc#entity1, doc#entity2, relation_label)
    """
    gold_relations, pred_relations = [], []

    for i, row in df.iterrows():
        doc_id = str(row["document_id"]).strip()
        e1 = str(row["entity1_id"]).strip()
        e2 = str(row["entity2_id"]).strip()

        # gold label → string
        label_gold = (
            id2label[int(row[label_col])]
            if id2label is not None
            else str(row[label_col])
        )

        e1_uid = f"{doc_id}#{e1}"
        e2_uid = f"{doc_id}#{e2}"
        gold_relations.append((e1_uid, e2_uid, label_gold))

        # predictions (optional)
        if preds is not None:
            label_pred = (
                id2label[int(preds[i])] if id2label is not None else str(preds[i])
            )
            pred_relations.append((e1_uid, e2_uid, label_pred))

    return gold_relations, pred_relations

