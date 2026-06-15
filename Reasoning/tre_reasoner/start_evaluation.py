import argparse
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

from data import TRECachedDataset, tre_collate_cached
from data_sampler import DocBatchSampler, SortedDocSampler
from model import create_model, BaseLine, ReasoningModel
from conf_loader import load_runtime_config, available_datasets
from evaluate import evaluate, evaluate_baseline, evaluate_baseline_reasoning
from eval_helpers import analyze_consistency, calculate_gt_energy, calculate_pred_energy

# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================
def run_evaluation(args):
    dataset_cfg, model_cfg = load_runtime_config(args.dataset, args.conf)
    if args.data_mode is not None:
        model_cfg.data_mode = args.data_mode
    if args.max_steps is not None:
        model_cfg.max_steps = args.max_steps
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve model_path from config default if not provided
    if args.model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        checkpoint_root = str(project_root / "artifacts" / "tre_reasoner" / "checkpoints")
        default_ckpt_name = "MATRES_baseline_reasoning_model.pt" if args.dataset.upper() == "MATRES" else "I2B2_baseline_reasoning_model.pt"
        args.model_path = os.path.join(checkpoint_root, default_ckpt_name)
    if args.split is None:
        args.split = "val"
    
    # 1. Load Data
    print("Loading Data...")
    if args.split == 'test':
        token_path = dataset_cfg.test_token_path
        jsonl_path = dataset_cfg.test_jsonl_path
    else:
        token_path = dataset_cfg.val_token_path
        jsonl_path = dataset_cfg.val_jsonl_path

    if (not token_path or not jsonl_path) and args.split == 'test':
        token_path = dataset_cfg.val_token_path
        jsonl_path = dataset_cfg.val_jsonl_path
    
    if not token_path or not jsonl_path:
        raise ValueError("No test/val data path found in config.")
        
    dataset = TRECachedDataset(token_path, jsonl_path)
    batch_size = args.batch_size if args.batch_size is not None else model_cfg.batch_size
    if model_cfg.data_mode == 'doc':
        sampler = DocBatchSampler(dataset.doc_ids, batch_size, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=tre_collate_cached
        )
    else:
        ordered_sampler = SortedDocSampler(dataset.doc_ids)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=ordered_sampler,
            collate_fn=tre_collate_cached
        )
    
    # 2. Prepare Model
    print("Initializing Model...")
    bert_model_name = args.bert_model if args.bert_model is not None else model_cfg.bert_model
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    bert = AutoModel.from_pretrained(bert_model_name)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    bert.resize_token_embeddings(len(tokenizer))
    
    if args.mode == 'hybrid':
        print("Mode: HYBRID (Baseline Weights + Strong Logic Inference)")
        model = create_model(
            mode='baseline_reasoning',
            bert=bert,
            num_classes=dataset_cfg.num_classes,
            rule_file_path=dataset_cfg.rule_file,
            relation_map=dataset_cfg.relation_map,
            freeze_bert=True,
            step_size=0.5,
            smooth_tau=0.01,
            max_steps=args.max_steps
        )
        
        if args.model_path:
            print(f"Loading Baseline weights from {args.model_path} into Encoder...")
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('bert.') or k.startswith('classifier.') or k.startswith('pair_proj.') or k.startswith('joint_layers.'):
                    new_state_dict[f"encoder.{k}"] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            
        with torch.no_grad():
            if hasattr(model.reasoning, "w_rules_logits"):
                model.reasoning.w_rules_logits.fill_(5.0)
            elif hasattr(model.reasoning, "rule_logits"):
                model.reasoning.rule_logits.fill_(5.0)
            
    elif args.mode == 'energy_gt' or args.mode == 'energy_pred':
        print("Mode: ENERGY CHECK (Ground Truth)")
        model = create_model(
            mode='baseline_reasoning',
            bert=bert,
            num_classes=dataset_cfg.num_classes,
            rule_file_path=dataset_cfg.rule_file,
            relation_map=dataset_cfg.relation_map,
            num_types=1,
            step_size=model_cfg.step_size,
            lambda_kl=model_cfg.lambda_kl,
            smooth_tau=model_cfg.smooth_tau,
            max_steps=model_cfg.max_steps,
            tol=model_cfg.tol,
            use_deq=model_cfg.use_deq,
            output_option=model_cfg.output_option,
            learn_rule_weights=model_cfg.learn_rule_weights,
            initial_rule_weight=model_cfg.initial_rule_weight,
            rule_chunk_size=model_cfg.rule_chunk_size,
            tokenizer=tokenizer
        )
        if args.model_path:
            print(f"Loading checkpoint {args.model_path}...")
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        
    else:
        if args.mode == 'baseline':
            model_mode = 'baseline'
        elif args.mode == 'reasoning':
            model_mode = 'reasoning'
        else:
            model_mode = 'baseline_reasoning'
            
        model = create_model(
            mode=model_mode,
            bert=bert,
            num_classes=dataset_cfg.num_classes,
            num_types=1,
            rule_file_path=dataset_cfg.rule_file,
            relation_map=dataset_cfg.relation_map,
            freeze_bert=args.freeze_bert if args.freeze_bert is not None else model_cfg.freeze_bert,
            step_size=model_cfg.step_size,
            lambda_kl=model_cfg.lambda_kl,
            smooth_tau=model_cfg.smooth_tau,
            max_steps=model_cfg.max_steps,
            tol=model_cfg.tol,
            use_deq=args.use_deq if args.use_deq is not None else model_cfg.use_deq,
            output_option=args.output_option if args.output_option is not None else model_cfg.output_option,
            # [MODIFIED] Match training config for correct graph construction
            learn_rule_weights=model_cfg.learn_rule_weights,
            initial_rule_weight=model_cfg.initial_rule_weight,
            rule_chunk_size=model_cfg.rule_chunk_size,
            tokenizer=tokenizer
        )
        if args.model_path:
             print(f"Loading checkpoint {args.model_path}...")
             ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
             if 'model_state_dict' in ckpt:
                 model.load_state_dict(ckpt['model_state_dict'], strict=False)
             else:
                 model.load_state_dict(ckpt, strict=False)

    model.to(device)
    
    if args.mode == 'energy_gt':
        avg_energy = calculate_gt_energy(
            model,
            dataloader,
            device,
            dataset_cfg,
            debug=args.debug_energy,
            max_samples=args.max_energy_samples,
            max_triplets=args.max_energy_triplets,
            energy_doc_id=args.energy_doc_id
        )
        print(f"Average Ground Truth Energy Violation: {avg_energy:.6f}")
        return
    if args.mode == 'energy_pred':
        avg_energy = calculate_pred_energy(
            model,
            dataloader,
            device,
            dataset_cfg,
            debug=args.debug_energy,
            use_q_as_pred=args.use_q_as_pred
        )
        print(f"Average Predicted Energy Violation: {avg_energy:.6f}")
        return

    model.eval()
    doc_predictions = defaultdict(dict)
    all_preds = []
    all_labels = []
    records = []
    id2label = {v: k for k, v in dataset_cfg.relation_map.items()} if dataset_cfg.relation_map else {0:'0',1:'1',2:'2',3:'3'}
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_marks = batch['word_marks'].to(device)
            e1_marks = batch['e1_marks'].to(device)
            e2_marks = batch['e2_marks'].to(device)
            labels = batch['labels'].to(device)
            
            doc_ids = batch['doc_ids']
            e1_ids = batch['e1_ids']
            e2_ids = batch['e2_ids']
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'word_marks': word_marks,
                'e1_marks': e1_marks,
                'e2_marks': e2_marks
            }
            
            if args.mode != 'baseline':
                 inputs.update({
                     'entity_pairs': list(zip(e1_ids, e2_ids)),
                     'doc_ids': doc_ids,
                     'e1_type_ids': batch['e1_type_ids'].to(device),
                     'e2_type_ids': batch['e2_type_ids'].to(device)
                 })
            
            out = model(**inputs)
            
            if isinstance(out, dict):
                logits_logic = out.get('L_F')
                logits_encoder = out.get('L_nn', out.get('L_F'))
                if args.use_q_as_pred:
                     q_probs = out.get('Q_T')
                     if q_probs is not None:
                         logits_logic = torch.log(q_probs + 1e-12)
            else:
                logits_logic = out
                logits_encoder = out
                
            preds_logic = torch.argmax(logits_logic, dim=1).cpu().numpy()
            preds_encoder = torch.argmax(logits_encoder, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds_logic)
            all_labels.extend(labels_np)
            
            for i in range(len(preds_logic)):
                doc_predictions[doc_ids[i]][(e1_ids[i], e2_ids[i])] = preds_logic[i]
                
                records.append({
                    "doc_id": doc_ids[i],
                    "e1_id": e1_ids[i],
                    "e2_id": e2_ids[i],
                    "ground_truth": id2label.get(labels_np[i], str(labels_np[i])),
                    "encoder_prediction": id2label.get(preds_encoder[i], str(preds_encoder[i])),
                    "prediction": id2label.get(preds_logic[i], str(preds_logic[i]))
                })

    print("\n[Predictive Performance]")

    def print_summary(y_true, y_pred, title):
        p_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        r_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        p_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        r_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"\n[{title} Summary]")
        print(f"Micro   : P={p_micro:.4f}, R={r_micro:.4f}, F1={f1_micro:.4f}")
        print(f"Macro   : P={p_macro:.4f}, R={r_macro:.4f}, F1={f1_macro:.4f}")
        print(f"Weighted: P={p_weighted:.4f}, R={r_weighted:.4f}, F1={f1_weighted:.4f}")

    all_label_ids = sorted(list(set(all_labels) | set(all_preds)))
    all_label_names = [id2label.get(i, str(i)) for i in all_label_ids]

    full_cm = confusion_matrix(all_labels, all_preds, labels=all_label_ids)
    print("\n[Confusion Matrix - Full] (rows=true, cols=pred)")
    print(pd.DataFrame(full_cm, index=all_label_names, columns=all_label_names))
    print("\n[Classification Report - Full (including VAGUE)]")
    print(classification_report(
        all_labels,
        all_preds,
        digits=4,
        labels=all_label_ids,
        target_names=all_label_names,
        zero_division=0
    ))
    print_summary(all_labels, all_preds, "Full (including VAGUE)")

    valid_indices = [i for i, x in enumerate(all_labels) if x != dataset_cfg.vague_label_id]
    if valid_indices:
        y_true = np.array(all_labels)[valid_indices]
        y_pred = np.array(all_preds)[valid_indices]
        nonv_cm = confusion_matrix(y_true, y_pred, labels=all_label_ids)
        print("\n[Confusion Matrix - Non-VAGUE Gold] (rows=true non-VAGUE, cols=pred)")
        print(pd.DataFrame(nonv_cm, index=all_label_names, columns=all_label_names))

        print("\n[Classification Report - Non-VAGUE Gold]")
        # [MODIFIED] Explicitly specify labels to valid classes (excluding VAGUE gold)
        valid_labels = sorted(list(set(y_true)))
        print(classification_report(
            y_true,
            y_pred,
            digits=4,
            labels=valid_labels,
            target_names=[id2label.get(i, str(i)) for i in valid_labels],
            zero_division=0
        ))
        print_summary(y_true, y_pred, "Non-VAGUE Gold")
    else:
        print("No valid samples for metric calculation (all VAGUE?).")

    if args.eval_consistency or args.mode in ['hybrid']:
        print("\n[Consistency Analysis]")
        # Pass dataset name and vague_id
        tvr, svr, n_trip, n_sym = analyze_consistency(
            doc_predictions, 
            dataset_name=dataset_cfg.name, 
            vague_id=dataset_cfg.vague_label_id
        )
        print(f"Transitivity Violation Rate: {tvr:.4f} (on {n_trip} triplets)")
        print(f"Symmetry Violation Rate:     {svr:.4f} (on {n_sym} pairs)")

    if args.output_file:
        df = pd.DataFrame(records)
        df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for TRE")
    parser.add_argument("--dataset", type=str, required=True, choices=available_datasets())
    parser.add_argument("--conf", type=str, default=None, help="Path to .conf file (optional). Defaults to tre_reasoner/configs/<DATASET>.conf")
    parser.add_argument("--mode", type=str, default="baseline", 
                        choices=["baseline", "baseline_reasoning", "reasoning", "hybrid", "energy_gt", "energy_pred"],
                        help="'hybrid' = baseline weights + strong rule inference")
    parser.add_argument("--model_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"], help="Evaluate on validation or test split")
    parser.add_argument("--bert_model", type=str, default=None)
    parser.add_argument("--data_mode", type=str, default=None, choices=["pair", "doc"], help="Data loading mode: pair or doc")
    parser.add_argument("--output_file", type=str, default=None, help="Save predictions to CSV")
    
    parser.add_argument("--eval_consistency", action="store_true", help="Calculate consistency metrics")
    parser.add_argument("--freeze_bert", action="store_true", help="If creating new model, freeze bert")
    parser.add_argument("--use_q_as_pred", action="store_true", help="Use Q (Reasoned Prob) for prediction instead of Final Logits")
    parser.add_argument("--output_option", type=str, default=None, help="Model output option")
    parser.add_argument("--use_deq", action="store_true", help="Use DEQ model")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max reasoning steps")
    parser.add_argument("--debug_energy", action="store_true", help="Print per-rule energy violation breakdown for GT energy")
    parser.add_argument("--max_energy_samples", type=int, default=3, help="Max GT-energy sample batches to log")
    parser.add_argument("--max_energy_triplets", type=int, default=10, help="Max GT-energy violating triplets to log")
    parser.add_argument("--energy_doc_id", type=str, default=None, help="Doc ID to log GT energy triplets for")
    
    args = parser.parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
