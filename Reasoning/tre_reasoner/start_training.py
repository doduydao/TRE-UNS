import argparse
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, BertTokenizerFast
from data import TRECachedDataset, tre_collate_cached
from data_sampler import DocBatchSampler
from model import create_model
from train import train_model
from config import get_runtime_configs, CONFIGS


def _resolve_doc_ids(dataset):
    """Return doc_ids for both TRECachedDataset and torch.utils.data.Subset wrappers."""
    if hasattr(dataset, "doc_ids"):
        return dataset.doc_ids
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "doc_ids"):
        return [dataset.dataset.doc_ids[i] for i in dataset.indices]
    raise AttributeError("Dataset does not expose doc_ids required for doc-mode batching")


def _resolve_labels(dataset):
    """Return labels for both TRECachedDataset and torch.utils.data.Subset wrappers."""
    if hasattr(dataset, "labels"):
        return dataset.labels
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "labels"):
        return dataset.dataset.labels[dataset.indices]
    raise AttributeError("Dataset does not expose labels")

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for TRE")
    parser.add_argument("--dataset", type=str, required=True, choices=list(CONFIGS.keys()), help="Dataset name (e.g., MATRES, I2B2)")
    parser.add_argument("--conf", type=str, default=None, help="Optional path to dataset .conf file")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "baseline_psl", "baseline_reasoning", "TRER"], help="Training mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    
    # Model Params (Override defaults)
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the model")
    
    # Debug Options
    parser.add_argument("--debug_subset", type=int, default=0, help="Subset size for debugging (e.g. 1000). 0 means full.")
    parser.add_argument("--debug_trajectory", action="store_true", help="Print detailed inference trajectory.")
    parser.add_argument("--symmetrize", action="store_true", help="Enable Test-Time Symmetrization (Logic-aware input).")
    parser.add_argument('--lambda_energy', type=float, default=None, help='Strength of logic constraints')
    parser.add_argument('--use_deq', action='store_true', help='Use Deep Equilibrium Model')
    parser.add_argument('--learn_rule_weights', action='store_true', help='Learn rule weights')
    parser.add_argument('--data_mode', type=str, default=None, choices=['pair', 'doc'], help='Data loading mode: pair or doc')
    parser.add_argument('--max_pairs_per_doc_batch', type=int, default=0, help='Max pairs per batch when using doc mode (0 means no cap)')

    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Performance flags (no structural changes)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Load Configs
    dataset_cfg, model_cfg = get_runtime_configs(args.dataset, conf_path=args.conf)
    
    # Overrides
    if args.epochs is not None: model_cfg.epochs = args.epochs
    if args.batch_size is not None: model_cfg.batch_size = args.batch_size
    if args.lr is not None: model_cfg.lr = args.lr
    if args.lambda_energy is not None: model_cfg.lambda_energy = args.lambda_energy
    if args.use_deq: model_cfg.use_deq = True
    if args.learn_rule_weights: model_cfg.learn_rule_weights = True
    if args.data_mode: model_cfg.data_mode = args.data_mode
    
    # Default Config Overrides specific to dataset if needed, 
    # but strictly we use command line or ModelConfig defaults.
    
    print(f"=== Configuration ===")
    print(f"Dataset: {dataset_cfg.name}")
    print(f"Mode: {args.mode}")
    print(f"Model: {model_cfg.bert_model}")
    print(f"Epochs: {model_cfg.epochs}")
    print(f"Batch Size: {model_cfg.batch_size}")
    print(f"Data Mode: {model_cfg.data_mode}")
    if model_cfg.data_mode == 'doc':
        print(f"Max Pairs/Doc Batch: {args.max_pairs_per_doc_batch}")
    
    # 1. Load Data
    print("\n[1/3] Loading Data...")
    
    # Check if files exist
    if not os.path.exists(dataset_cfg.train_token_path):
        raise FileNotFoundError(f"Train cache not found: {dataset_cfg.train_token_path}")
        
    train_dataset = TRECachedDataset(dataset_cfg.train_token_path, dataset_cfg.train_jsonl_path)
    val_dataset = TRECachedDataset(dataset_cfg.val_token_path, dataset_cfg.val_jsonl_path)
    
    # Subset if requested
    if args.debug_subset > 0:
        print(f"[DEBUG] Subsetting train/val data to {args.debug_subset} samples.")
        train_indices = list(range(min(args.debug_subset, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        val_indices = list(range(min(args.debug_subset, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_labels = _resolve_labels(train_dataset)
    train_label_counter = Counter(train_labels.cpu().tolist())
    label_name_by_id = {v: k for k, v in dataset_cfg.relation_map.items()} if dataset_cfg.relation_map else {}
    print("[Train Label Distribution]")
    for label_id in sorted(train_label_counter.keys()):
        label_name = label_name_by_id.get(label_id, str(label_id))
        print(f"  - {label_name}({label_id}): {train_label_counter[label_id]}")
    if dataset_cfg.vague_label_id is not None:
        vague_count = train_label_counter.get(dataset_cfg.vague_label_id, 0)
        print(f"[Train] VAGUE label count: {vague_count} (included in training)")
    
    if model_cfg.data_mode == 'doc':
        max_pairs = args.max_pairs_per_doc_batch if args.max_pairs_per_doc_batch > 0 else None
        train_doc_ids = _resolve_doc_ids(train_dataset)
        val_doc_ids = _resolve_doc_ids(val_dataset)
        train_sampler = DocBatchSampler(
            train_doc_ids,
            model_cfg.batch_size,
            shuffle=True,
            max_pairs_per_batch=max_pairs,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=tre_collate_cached,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_sampler = DocBatchSampler(
            val_doc_ids,
            model_cfg.batch_size,
            shuffle=False,
            max_pairs_per_batch=max_pairs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=tre_collate_cached,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        # Pair-level batching (no document-level sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_cfg.batch_size,
            shuffle=True,
            collate_fn=tre_collate_cached,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_cfg.batch_size,
            shuffle=False,
            collate_fn=tre_collate_cached,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    # 2. Setup BERT & Model
    print("\n[2/3] Initializing Model...")
    tokenizer = BertTokenizerFast.from_pretrained(model_cfg.bert_model)
    bert = AutoModel.from_pretrained(model_cfg.bert_model)
    
    # Resize for special tokens
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    bert.resize_token_embeddings(len(tokenizer))
    
    # Create Model
    model = create_model(
        mode=args.mode,
        bert=bert,
        num_classes=dataset_cfg.num_classes,
        num_types=1,
        rule_file_path=dataset_cfg.rule_file,
        relation_map=dataset_cfg.relation_map,
        freeze_bert=model_cfg.freeze_bert, # Currently False in config
        step_size=model_cfg.step_size,
        lambda_kl=model_cfg.lambda_kl,
        smooth_tau=model_cfg.smooth_tau,
        lambda_entropy=getattr(model_cfg, "lambda_entropy", 0.0),
        max_steps=model_cfg.max_steps,
        tol=model_cfg.tol,
        use_deq=model_cfg.use_deq,
        output_option=model_cfg.output_option,
        tokenizer=tokenizer, # [ADDED] Pass tokenizer for special ID access
        learn_rule_weights=model_cfg.learn_rule_weights,
        initial_rule_weight=model_cfg.initial_rule_weight,
        rule_chunk_size=model_cfg.rule_chunk_size,
    )
    
    # 3. Train
    print("\n[3/3] Starting Training...")
    
    if args.save_path:
        save_file = args.save_path
    else:
        save_file = f"checkpoints/{dataset_cfg.name}_{args.mode}_model.pt"
        os.makedirs("checkpoints", exist_ok=True)
        
    train_model(
        mode=args.mode,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=model_cfg.epochs,
        lr=model_cfg.lr,
        save_path=save_file,
        lambda_distill=model_cfg.lambda_distill,
        lambda_energy=model_cfg.lambda_energy,
        use_amp=model_cfg.use_amp,
        max_grad_norm=model_cfg.max_grad_norm,
        class_weights=dataset_cfg.class_weights,
        vague_label_id=dataset_cfg.vague_label_id,
        is_MATRES=(args.dataset=="MATRES"),
        is_I2B2=(args.dataset=="I2B2"),
        is_TBD=(args.dataset=="TBD"),
        is_TDD=(args.dataset=="TDD" or args.dataset=="TDDMAN")
    )

if __name__ == "__main__":
    main()
