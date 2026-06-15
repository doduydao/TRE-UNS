import argparse
import os
import multiprocessing
import spacy
from spacy.util import compile_infix_regex
from spacy.tokenizer import Tokenizer
from transformers import AutoTokenizer

from data import build_spacy_and_token_cache, loadi2b2, loadMATRES
from config import get_config, CONFIGS

def create_custom_tokenizer(nlp, special_tokens):
    infix_re = compile_infix_regex(nlp.Defaults.infixes)
    custom_tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)
    special_case = {token: [{spacy.symbols.ORTH: token}] for token in special_tokens}
    for token in special_tokens:
        custom_tokenizer.add_special_case(token, special_case[token])
    return custom_tokenizer

def run_preprocessing(args):
    dataset_cfg = get_config(args.dataset)
    print(f"Preprocessing {args.dataset}...")
    
    # 1. Load Raw Data
    if args.dataset == 'I2B2':
        data_info = loadi2b2()
    elif args.dataset == 'MATRES':
        data_info = loadMATRES()
    else:
        raise ValueError("Unknown dataset")
        
    # 2. Setup Tools
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_nlp.tokenizer = create_custom_tokenizer(spacy_nlp, special_tokens)
    
    n_proc = max(1, multiprocessing.cpu_count() - 1)
    
    # 3. Build Caches
    tasks = [
        ('train', dataset_cfg.train_jsonl_path, dataset_cfg.train_token_path, data_info['train_df']),
        ('valid', dataset_cfg.val_jsonl_path, dataset_cfg.val_token_path, data_info['valid_df']),
        ('test', dataset_cfg.test_jsonl_path, dataset_cfg.test_token_path, data_info['test_df'])
    ]
    
    for name, jsonl_path, token_path, df in tasks:
        if not jsonl_path or not token_path or df is None:
            print(f"Skipping {name} (config missing path or data None)")
            continue
            
        # Ensure dir exists
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        
        print(f"\nBuilding {name} cache for {args.dataset}...")
        build_spacy_and_token_cache(
            df=df,
            spacy_nlp=spacy_nlp,
            tokenizer=tokenizer,
            window_size=args.window_size,
            max_length=args.max_length,
            jsonl_path=jsonl_path,
            token_pt=token_path,
            batch_size=args.batch_size,
            n_process=n_proc
        )

def main():
    parser = argparse.ArgumentParser(description="Unified Preprocessing Script")
    parser.add_argument("--dataset", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    run_preprocessing(args)

if __name__ == "__main__":
    main()
