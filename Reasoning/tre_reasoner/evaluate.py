import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix
)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from i2b2_eval import build_relations_from_df, evaluate_tempeval3

def evaluate(
    dataloader,
    model,
    top_k=1, # Default
    device=None,
    threshold=1.0,
    return_preds=False,
    return_topk_tokens=False,
    return_logs=False,
    is_I2B2=False,
    is_TDD=False,
    is_TBD=False,
    is_MATRES=False,
    vague_label_id=None,
    test_df=None,
    id2label=None,
):
    """
    Evaluate model for different temporal datasets.
    - I2B2: closure-based if test_df provided, else weighted F1.
    - TDD/TBD/MATRES: micro-F1.
    - Logs & predictions can be optionally returned.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    start_time = time.time()

    all_predictions = []
    all_labels = []
    logs = []
    skipped_vague = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            def _filter_by_indices(x, keep_idx_cpu, keep_idx_t):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return x[keep_idx_t]
                return [x[i] for i in keep_idx_cpu]

            # Handle variable input keys based on model type
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            total_samples += labels.size(0)

            # Optional args
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            # Additional args for Reasoning/Baseline model
            e1_marks = batch.get('e1_marks')
            if e1_marks is not None: e1_marks = e1_marks.to(device)
            
            e2_marks = batch.get('e2_marks')
            if e2_marks is not None: e2_marks = e2_marks.to(device)
            
            word_marks = batch.get('word_marks')
            if word_marks is not None: word_marks = word_marks.to(device)

            e1_type_ids = batch.get('e1_type_ids')
            if e1_type_ids is not None: e1_type_ids = e1_type_ids.to(device)

            e2_type_ids = batch.get('e2_type_ids')
            if e2_type_ids is not None: e2_type_ids = e2_type_ids.to(device)
            
            doc_ids = batch.get('doc_ids')
            e1_ids_batch = batch.get('e1_ids')
            e2_ids_batch = batch.get('e2_ids')

            # Exclude VAGUE-labeled pairs before forward during evaluation.
            if vague_label_id is not None:
                keep_mask = (labels != vague_label_id)
                skipped_vague += (~keep_mask).sum().item()
                if keep_mask.sum().item() == 0:
                    continue

                keep_idx_t = keep_mask.nonzero(as_tuple=False).squeeze(-1)
                keep_idx_cpu = keep_idx_t.detach().cpu().tolist()

                input_ids = input_ids[keep_idx_t]
                attention_mask = attention_mask[keep_idx_t]
                labels = labels[keep_idx_t]

                token_type_ids = _filter_by_indices(token_type_ids, keep_idx_cpu, keep_idx_t)
                e1_marks = _filter_by_indices(e1_marks, keep_idx_cpu, keep_idx_t)
                e2_marks = _filter_by_indices(e2_marks, keep_idx_cpu, keep_idx_t)
                word_marks = _filter_by_indices(word_marks, keep_idx_cpu, keep_idx_t)
                e1_type_ids = _filter_by_indices(e1_type_ids, keep_idx_cpu, keep_idx_t)
                e2_type_ids = _filter_by_indices(e2_type_ids, keep_idx_cpu, keep_idx_t)
                doc_ids = _filter_by_indices(doc_ids, keep_idx_cpu, keep_idx_t)
                e1_ids_batch = _filter_by_indices(e1_ids_batch, keep_idx_cpu, keep_idx_t)
                e2_ids_batch = _filter_by_indices(e2_ids_batch, keep_idx_cpu, keep_idx_t)

            if isinstance(model, nn.Module) and hasattr(model, 'reasoning'):
                # Reasoning Model Call
                 entity_pairs = list(zip(e1_ids_batch, e2_ids_batch))
                 if isinstance(doc_ids, torch.Tensor): doc_ids = doc_ids.tolist()
                 
                 out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_marks=word_marks,
                    e1_marks=e1_marks,
                    e2_marks=e2_marks,
                    entity_pairs=entity_pairs, 
                    doc_ids=doc_ids, 
                    e1_type_ids=e1_type_ids, 
                    e2_type_ids=e2_type_ids
                 )
                 # Output is dict
                 logits = out['L_F'] # Logits Final
                 
            elif isinstance(model, nn.Module) and (hasattr(model, 'encoder') or hasattr(model, 'bert')): 
                 # Baseline or similar (has 'bert' attribute)
                 out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_marks=word_marks,
                    e1_marks=e1_marks,
                    e2_marks=e2_marks
                 )
                 if isinstance(out, dict):
                     logits = out['L_F']
                 else:
                     logits = out
            else:
                 # Fallback for generic forward call (e.g. from user snippet context?)
                 # Use kwargs or assume same signature
                 out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_marks=word_marks, 
                    e1_marks=e1_marks, 
                    e2_marks=e2_marks
                 )
                 if isinstance(out, dict):
                    logits = out['L_F'] if 'L_F' in out else list(out.values())[0]
                 else:
                    logits = out

            batch_logs = None
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())


    elapsed_time = time.time() - start_time
    print(f"Evaluation finished in {elapsed_time:.2f}s.")
    if vague_label_id is not None:
        print(f"Filtered {skipped_vague}/{total_samples} "
              f"({100*skipped_vague/total_samples:.2f}%) VAGUE samples.")

    # ==================================================
    # ===  I2B2 EVALUATION
    # ==================================================
    if is_I2B2:
        if test_df is not None:
            print("Using closure-based evaluation for I2B2 (document-level).")
            gold_relations, pred_relations = build_relations_from_df(
                test_df,
                preds=all_predictions,
                label_col="label_encoded",
                id2label=id2label
            )
           
            metrics = evaluate_tempeval3(gold_relations, pred_relations, use_fast=True)
            F1, P, R = metrics["F1"], metrics["Precision"], metrics["Recall"]
            n_gold, n_pred, n_gold_closure, n_pred_closure = metrics["Gold"],  metrics["Pred"],  metrics["GoldClosure"],  metrics["PredClosure"]
        
            print("====== [I2B2 CLOSURE-BASED EVALUATION] ======")
            print(f"Precision: {P:.4f}")
            print(f"Recall:    {R:.4f}")
            print(f"F1-score:  {F1:.4f}")
            print("==============================================")

            if return_preds or return_logs:
                return (None, F1, P, R, [], [], [], all_predictions, all_labels, logs)
            else:
                return None, F1, P, R, [], [], []
        else:
            print("Using standard weighted-F1 evaluation for I2B2 (no test_df provided).")
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            p = precision_score(all_labels, all_predictions, average='weighted')
            r = recall_score(all_labels, all_predictions, average='weighted')
            cm = confusion_matrix(all_labels, all_predictions)
            precisions, recalls, f1s, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
            print(f"[I2B2] F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
            
            if return_preds or return_logs:
                return (cm, f1, p, r, precisions, recalls, f1s, all_predictions, all_labels, logs)
            else:
                return cm, f1, p, r, precisions, recalls, f1s

    if is_TBD or is_MATRES:
        f1 = f1_score(all_labels, all_predictions, average='micro')
        p = precision_score(all_labels, all_predictions, average='micro')
        r = recall_score(all_labels, all_predictions, average='micro')
        cm = confusion_matrix(all_labels, all_predictions)
        precisions, recalls, f1s, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        print(f"[MATRES/TBD] Micro-F1={f1:.4f}, P={p:.4f}, R={r:.4f}")

        if return_preds or return_logs:
            return (cm, f1, p, r, precisions, recalls, f1s, all_predictions, all_labels, logs)
        else:
            return cm, f1, p, r, precisions, recalls, f1s
    if is_TDD:
        cm = confusion_matrix(all_labels, all_predictions)
        F1 = f1_score(all_labels, all_predictions, average='micro')
        P = precision_score(all_labels, all_predictions, average='micro')
        R = recall_score(all_labels, all_predictions, average='micro')
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        print(f"[TDD] Micro-F1={F1:.4f}, P={P:.4f}, R={R:.4f}")

        if return_preds or return_logs:
            return (
                cm, F1, P, R, precisions, recalls, f1s,
                all_predictions, all_labels, logs
            )
        else:
            return cm, F1, P, R, precisions, recalls, f1s
    
    # ==================================================
    # === DEFAULT CASE (NO FLAGS)
    # ==================================================
    if return_preds or return_logs:
        return None, 0, 0, 0, [], [], [], all_predictions, all_labels, logs
    else:
        return None, 0, 0, 0, [], [], []


def evaluate_baseline(model, dataloader, device=None):
    if device is None: device = next(model.parameters()).device
    return evaluate(dataloader, model, device=device, is_MATRES=True, vague_label_id=3) # Assume MATRES

def evaluate_baseline_reasoning(model, dataloader, device=None):
    if device is None: device = next(model.parameters()).device
    return evaluate(dataloader, model, device=device, is_MATRES=True, vague_label_id=3) 

def evaluate_and_show(dataloader,
                      model,
                      top_k,
                      device,
                      relations,
                      return_preds=False,
                      is_I2B2=False,
                      is_TDD=False,
                      is_TBD=False,
                      is_MATRES=False,
                      vague_label_id=None,
                      test_df=None):
    cm, f1, p, r, precisions, recalls, f1s = evaluate(
        dataloader,
        model=model,
        top_k=top_k,
        device=device,
        return_preds=return_preds,
        is_I2B2=is_I2B2,
        is_TDD=is_TDD,
        is_TBD=is_TBD,
        is_MATRES=is_MATRES,
        vague_label_id=vague_label_id,
        test_df=test_df
    )

    print(f"\nEval Summary: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
    if relations and len(relations) == len(precisions):
        for i in range(len(relations)):
            print(f"{relations[i]:<15}: F1={f1s[i]:.2f}, P={precisions[i]:.2f}, R={recalls[i]:.2f}")
    print("Confusion Matrix:")
    print(cm)

def grid_search(data_loader,
                model,
                K,
                device,
                is_MATRES=False,
                is_I2B2=False,
                is_TBD=False,
                is_TDD=False,
                vague_label_id=None):
    metrics_by_topk = []
    for k in range(0,K+1,1):
        cm, f1, p, r, precisions,recalls, f1s = evaluate(data_loader,
                                                        model=model,
                                                        top_k=k,
                                                        device=device,
                                                        return_preds=False,
                                                        is_MATRES=is_MATRES,
                                                        is_I2B2=is_I2B2,
                                                        is_TBD=is_TBD,
                                                        is_TDD=is_TDD,
                                                        vague_label_id=vague_label_id)
        metrics_by_topk.append([k, round(f1, 4), round(p,4), round(r, 4)])
        print(f"top_K: {k} - F1: {f1:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")
        print()
    return metrics_by_topk

def predict_and_save(model, dataloader, output_file, id2label, device=None):
     pass