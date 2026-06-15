#!/usr/bin/env python3
"""
Resume training from checkpoint for additional epochs
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast
from data import TRECachedDataset, tre_collate_cached
from model import create_model
from train import train_model
from config import get_config, ModelConfig
import os

def main():
    # Config
    dataset_name = 'MATRES'
    additional_epochs = 10
    checkpoint_path = 'checkpoints/MATRES_baseline_reasoning_model.pt'
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Load configs
    dataset_cfg = get_config(dataset_name)
    model_cfg = ModelConfig()
    
    print(f"=== Resuming Training ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Additional Epochs: {additional_epochs}")
    print(f"Current config: lambda_kl={model_cfg.lambda_kl}, output_option={model_cfg.output_option}")
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_cfg.bert_model)
    special_tokens = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load BERT
    bert = AutoModel.from_pretrained(model_cfg.bert_model)
    bert.resize_token_embeddings(len(tokenizer))
    
    # Create model
    model = create_model(
        mode='baseline_reasoning',
        bert=bert,
        num_classes=dataset_cfg.num_classes,
        rule_file_path=dataset_cfg.rule_file,
        relation_map=dataset_cfg.relation_map,
        freeze_bert=model_cfg.freeze_bert,
        tokenizer=tokenizer,
        step_size=model_cfg.step_size,
        lambda_kl=model_cfg.lambda_kl,
        smooth_tau=model_cfg.smooth_tau,
        max_steps=model_cfg.max_steps,
        tol=model_cfg.tol,
        use_deq=model_cfg.use_deq,
        output_option=model_cfg.output_option
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1  # Resume from next epoch
    print(f"Loaded checkpoint from epoch {start_epoch - 1}")
    
    # Load data
    print("Loading datasets...")
    train_ds = TRECachedDataset(dataset_cfg.train_token_path, dataset_cfg.train_jsonl_path)
    val_ds = TRECachedDataset(dataset_cfg.val_token_path, dataset_cfg.val_jsonl_path)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        collate_fn=tre_collate_cached,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg.batch_size,
        shuffle=False,
        collate_fn=tre_collate_cached,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    # Train (train_model creates optimizer internally)
    print(f"\n=== Training for {additional_epochs} more epochs ===\n")
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=additional_epochs,
        lr=model_cfg.lr,
        max_grad_norm=model_cfg.max_grad_norm,
        use_amp=model_cfg.use_amp,
        lambda_distill=model_cfg.lambda_distill,
        lambda_energy=model_cfg.lambda_energy,
        vague_label_id=dataset_cfg.vague_label_id,
        save_path=checkpoint_path,
        resume_from=checkpoint_path,  # Resume from checkpoint
        symmetrize_logits=False,
        warmup_epochs=0,  # No warmup for resumed training
        debug_trajectory=False
    )
    
    print(f"\n=== Training Complete ===")
    print(f"Model saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()
