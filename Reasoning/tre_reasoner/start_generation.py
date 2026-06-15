import argparse
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast
from tqdm import tqdm
from collections import defaultdict

from data import TRECachedDataset, tre_collate_cached
from model import create_model
from config import get_config, CONFIGS

def load_weights(model, path, device):
    print(f"Loading weights from {path}...")
    try:
        if torch.cuda.is_available():
            ckpt = torch.load(path, weights_only=False)
        else:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
        else:
             # Try loading as full model? Rare.
             model.load_state_dict(ckpt.state_dict())
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e

def generate(args):
    dataset_cfg = get_config(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading Data for {dataset_cfg.name}...")
    token_path = dataset_cfg.test_token_path if dataset_cfg.test_token_path and os.path.exists(dataset_cfg.test_token_path) else dataset_cfg.val_token_path
    jsonl_path = dataset_cfg.test_jsonl_path if dataset_cfg.test_jsonl_path and os.path.exists(dataset_cfg.test_jsonl_path) else dataset_cfg.val_jsonl_path
    
    if not token_path or not jsonl_path:
        raise ValueError(f"No test/val data found for {args.dataset}")
        
    dataset = TRECachedDataset(token_path, jsonl_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=tre_collate_cached)
    
    # 2. Setup Model
    print(f"Initializing {args.mode} model...")
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)
    bert = AutoModel.from_pretrained(args.bert_model)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    bert.resize_token_embeddings(len(tokenizer))
    
    # Reasoning Params Overrides
    step_size = args.step_size if args.step_size is not None else 0.5
    max_steps = args.max_steps if args.max_steps is not None else 50
    lambda_kl = args.lambda_kl if args.lambda_kl is not None else 0.1
    
    model = create_model(
        mode=args.mode,
        bert=bert,
        num_classes=dataset_cfg.num_classes,
        rule_file_path=dataset_cfg.rule_file,
        relation_map=dataset_cfg.relation_map,
        freeze_bert=args.freeze_bert, # Often True for inference optimization
        step_size=step_size,
        max_steps=max_steps,
        lambda_kl=lambda_kl,
        output_option=args.output_option
    )
    
    # Load Model Path
    if args.model_path:
        load_weights(model, args.model_path, device)
        
    model.to(device)
    model.eval()
    
    # 3. Label Map
    # Invert relation map if exists, else infer standard
    if dataset_cfg.relation_map:
        id2label = {v: k for k, v in dataset_cfg.relation_map.items()}
    else:
        # Fallback for I2B2 if not in config explicitly (usually 0:BEFORE, 1:AFTER, 2:OVERLAP based on alphabetical, OR check previous file)
        # generate_i2b2_baseline.py used: {0: 'AFTER', 1: 'BEFORE', 2: 'OVERLAP'} inferred alphabetical?
        # Actually generate_i2b2_baseline.py used {0: 'AFTER', 1: 'BEFORE', 2: 'OVERLAP'} at the end.
        # Let's stick to providing a mapping or defaulting.
        if args.dataset == "I2B2":
             id2label = {0: 'AFTER', 1: 'BEFORE', 2: 'OVERLAP'}
        else:
             id2label = {i: str(i) for i in range(dataset_cfg.num_classes)}
             
    print(f"Using Label Map: {id2label}")
    
    # 4. Inference
    records = []
    print("Generating Predictions...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_marks = batch['word_marks'].to(device)
            e1_marks = batch['e1_marks'].to(device)
            e2_marks = batch['e2_marks'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Metadata
            doc_ids = batch['doc_ids']
            e1_ids = batch['e1_ids']
            e2_ids = batch['e2_ids']
            
            # Forward
            if args.mode == 'baseline':
                out = model(input_ids, attention_mask, word_marks, e1_marks, e2_marks)
                logits = out['LF'] if isinstance(out, dict) else out
            else:
                 # Reasoning
                 out = model(
                     input_ids=input_ids,
                     attention_mask=attention_mask, 
                     word_marks=word_marks, 
                     e1_marks=e1_marks, 
                     e2_marks=e2_marks,
                     entity_pairs=list(zip(e1_ids, e2_ids)),
                     doc_ids=doc_ids,
                     e1_type_ids=batch['e1_type_ids'].to(device),
                     e2_type_ids=batch['e2_type_ids'].to(device)
                 )
                 logits = out['LF']
                 if args.use_q_as_pred and 'RP' in out:
                     logits = torch.log(out['RP'] + 1e-12)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(preds)):
                records.append({
                    "doc_id": doc_ids[i],
                    "E1_id": e1_ids[i],
                    "E2_id": e2_ids[i],
                    "Ground-Truth": id2label.get(labels[i], "UNKNOWN"),
                    "Prediction": id2label.get(preds[i], "UNKNOWN")
                })
                
    # 5. Save
    df = pd.DataFrame(records)
    if args.output_file:
        out_path = args.output_file
    else:
        out_path = f"predictions_{args.dataset}_{args.mode}.csv"
        
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path} ({len(df)} records)")


def main():
    parser = argparse.ArgumentParser(description="Unified Generation Script for TRE")
    parser.add_argument("--dataset", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "baseline_reasoning", "reasoning"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--freeze_bert", action="store_true")
    
    # Reasoning Specific
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lambda_kl", type=float, default=None)
    parser.add_argument("--output_option", type=str, default="direct_q")
    parser.add_argument("--use_q_as_pred", action="store_true")

    args = parser.parse_args()
    generate(args)

if __name__ == "__main__":
    main()
