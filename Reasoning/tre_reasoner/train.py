import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
import math
from collections import defaultdict
from evaluate import evaluate  # Generic evaluate
from reasoning_v2 import UnifiedNeuralReasoningLayerV2
import numpy as np


def _setup_device(model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)
    return device


def _get_optimizer(model, lr, w_rules_lr=None):
    """
    Create separate optimizers for rule weights and neural network parameters.
    
    Args:
        model: The model
        lr: Learning rate for neural network (BERT + classifier)
        w_rules_lr: Learning rate for rule weights (default: lr * 5)
    
    Returns:
        tuple: (nn_optimizer, w_rules_optimizer)
    """
    if w_rules_lr is None:
        w_rules_lr = lr  # Same LR as NN (was lr * 5.0, too high!)
    
    # Separate w_rules_logits from other parameters
    w_rules_params = []
    nn_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'w_rules_logits' in name:
            w_rules_params.append(param)
        else:
            nn_params.append(param)
    
    # Neural network optimizer (BERT + classifier)
    # Standard AdamW with weight decay
    nn_optimizer = torch.optim.AdamW(
        nn_params,
        lr=lr,
        weight_decay=0.01
    )
    
    # Rule weights optimizer
    # Higher LR, NO weight decay (we want weights to grow for violated rules)
    if len(w_rules_params) > 0:
        w_rules_optimizer = torch.optim.AdamW(
            w_rules_params,
            lr=w_rules_lr,
            weight_decay=0.0  # No decay - allow unbounded growth
        )
    else:
        w_rules_optimizer = None
    
    print(f"[Optimizer] Neural Network LR: {lr:.2e}, Rule Weights LR: {w_rules_lr:.2e}")
    print(f"[Optimizer] NN params: {len(nn_params)}, Rule params: {len(w_rules_params)}")
    
    return nn_optimizer, w_rules_optimizer


def _load_checkpoint(model, nn_optimizer, w_rules_optimizer, resume_path, device):
    """Load checkpoint for resuming training with dual optimizers."""
    best_val_f1 = 0.0
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizers if available
            if 'nn_optimizer_state_dict' in checkpoint and nn_optimizer:
                nn_optimizer.load_state_dict(checkpoint['nn_optimizer_state_dict'])
            if 'w_rules_optimizer_state_dict' in checkpoint and w_rules_optimizer:
                w_rules_optimizer.load_state_dict(checkpoint['w_rules_optimizer_state_dict'])
            # Backward compatibility: if old checkpoint has single optimizer
            elif 'optimizer_state_dict' in checkpoint and nn_optimizer:
                nn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            print(f"Resumed from epoch {start_epoch} with best val F1 {best_val_f1:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights only (legacy checkpoint). Starting from epoch 0.")
    return start_epoch, best_val_f1


def _save_checkpoint(model, nn_optimizer, w_rules_optimizer, epoch, val_f1, best_val_f1, save_path):
    """Save checkpoint with dual optimizers."""
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        # Prepare state dict safely
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'nn_optimizer_state_dict': nn_optimizer.state_dict(),
            'best_val_f1': best_val_f1,
        }
        if w_rules_optimizer is not None:
            checkpoint_dict['w_rules_optimizer_state_dict'] = w_rules_optimizer.state_dict()
            
        torch.save(checkpoint_dict, save_path)
        print(f"  Saved best model to {save_path} (F1: {val_f1:.4f})")
    return best_val_f1
# ============================================================
# 2. UNIFIED TRAINING FUNCTION
# ============================================================

def _extract_info_from_history(energy_history, info_type='all'):
    """
    Extract specific info from energy_history snapshots.
    
    Args:
        energy_history: List of energy snapshots from infer_Q
        info_type: 'P', 'Q', 'all', or specific field name
        
    Returns:
        dict: Extracted information
    """
    if isinstance(energy_history, dict):
        energy_history = energy_history.get('energy_history', [])

    if not energy_history or len(energy_history) == 0:
        return {}

    if len(energy_history) == 1 and isinstance(energy_history[0], list):
        energy_history = energy_history[0]

    P_snap = energy_history[0]  # Initial snapshot
    if isinstance(P_snap, list):
        P_snap = P_snap[0] if len(P_snap) > 0 else {}
         
    Q_snap = energy_history[-1]  # Final snapshot
    if isinstance(Q_snap, list):
        Q_snap = Q_snap[0] if len(Q_snap) > 0 else {}

    def _get_field(snap, key, default):
        return snap.get(key, default) if isinstance(snap, dict) else default

    def _get_energy_re(snap):
        return _get_field(snap, 'E_re', 0.0)

    def _get_energy_rules(snap):
        return _get_field(snap, 'E_rules', 0.0)

    def _get_groundings(snap):
        return _get_field(snap, 'GR', 0)

    def _get_violations(snap):
        return _get_field(snap, 'VC', 0)
    
    if info_type == 'P':
        return {
            'E_re': _get_energy_re(P_snap),
            'VC': _get_violations(P_snap),
            'GR': _get_groundings(P_snap),
            'rule_details': _get_field(P_snap, 'rule_details', {})
        }
    elif info_type == 'Q':
        return {
            'E_re': _get_energy_re(Q_snap),
            'VC': _get_violations(Q_snap),
            'GR': _get_groundings(Q_snap),
            'rule_details': _get_field(Q_snap, 'rule_details', {})
        }
    else:  # 'all'
        return {
            'E_re_P': _get_energy_re(P_snap),
            'E_re_Q': _get_energy_re(Q_snap),
            'E_rules_P': _get_energy_rules(P_snap),
            'E_rules_Q': _get_energy_rules(Q_snap),
            'VC_P': _get_violations(P_snap),
            'VC_Q': _get_violations(Q_snap),
            'GR': _get_groundings(Q_snap),
            'rule_details_P': _get_field(P_snap, 'rule_details', {}),
            'rule_details_Q': _get_field(Q_snap, 'rule_details', {})
        }


def _extract_reasoning_stats(out):
    """
    Extract reasoning statistics from model output.
    
    Args:
        out: Model output dictionary
        
    Returns:
        dict: Dictionary containing extracted stats
    """
   
    if not isinstance(out, dict):
        return {
            'L_F': out.get('L_F'),
            'P_F': None,
            'Q_T': None,
            'L_nn': None,
            'P': None,
            'E_re_P': 0.0,
            'E_re_Q': 0.0,
            'E_rules_P': 0.0,
            'E_rules_Q': 0.0,
            'VC_P': 0,
            'VC_Q': 0,
            'GR': 0.0,
            'rule_details_Q': {},
            'rule_details_P': {},
            'T': 0
        }

    # Extract basic outputs (align with model.py)
    L_F = out.get('L_F')
    P_F = out.get('P_F')
    Q_T = out.get('Q_T')
    L_nn = out.get('L_nn')
    P = out.get('P')

    # Extract stats from extras
    extras = out.get('extras', {})
    psl_stats = extras.get('psl_stats', None)
    if psl_stats is not None:
        E_rules = psl_stats.get('E_rules', 0.0)
        GR = psl_stats.get('GR', 0.0)
        VC = psl_stats.get('VC', 0.0)
        rule_details = psl_stats.get('rule_details', {})
        return {
            'L_F': L_F,
            'P_F': P_F,
            'Q_T': None,
            'L_nn': L_nn,
            'P': P,
            'E_re_P': E_rules,
            'E_re_Q': E_rules,
            'E_rules_P': E_rules,
            'E_rules_Q': E_rules,
            'VC_P': VC,
            'VC_Q': VC,
            'GR': GR,
            'rule_details_Q': rule_details,
            'rule_details_P': rule_details,
            'T': 1,
        }

    energy_history = extras.get('energy_history', [])
    if isinstance(energy_history, dict):
        energy_history = energy_history.get('energy_history', [])
    if len(energy_history) == 1 and isinstance(energy_history[0], list):
        energy_history = energy_history[0]
    
    # Use helper to extract from energy_history
    if len(energy_history) > 0:
        info = _extract_info_from_history(energy_history, 'all')
        E_re_P = info['E_re_P']
        E_re_Q = info['E_re_Q']
        E_rules_P = info['E_rules_P']
        E_rules_Q = info['E_rules_Q']
        VC_P = info['VC_P']
        VC_Q = info['VC_Q']
        GR = info['GR']
        rule_details_Q = info['rule_details_Q']
        rule_details_P = info['rule_details_P']
        total_steps = energy_history[-1].get('step', 0)
    else:
        # Fallback (shouldn't happen with refactored code)
        E_re_P = 0.0
        E_re_Q = 0.0
        E_rules_P = 0.0
        E_rules_Q = 0.0
        VC_P = 0
        VC_Q = 0
        GR = 0.0
        rule_details_Q = {}
        rule_details_P = {}
        total_steps = 0
    
    
    return {
        'L_F': L_F,
        'P_F': P_F,
        'Q_T': Q_T,
        'L_nn': L_nn,
        'P': P,
        'E_re_P': E_re_P,
        'E_re_Q': E_re_Q,
        'E_rules_P': E_rules_P,
        'E_rules_Q': E_rules_Q,
        'VC_P': VC_P,
        'VC_Q': VC_Q,
        'GR': GR,
        'rule_details_Q': rule_details_Q,
        'rule_details_P': rule_details_P,
        'T': total_steps,
    }


def _compute_losses(labels,
                    L_F,
                    Q_T,
                    P,
                    E_rules_P,
                    loss_fn,
                    kl_div, 
                    lambda_distill,
                    lambda_energy,
                    vague_label_id,
                    device,
                    mode):
    ce_loss = loss_fn(L_F, labels)
    
    
    if mode == 'baseline':
        kl_loss = torch.tensor(0.0, device=device)
        energy_loss = torch.tensor(0.0, device=device)
        total_loss = ce_loss
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'energy_loss': energy_loss,
        }

    if mode == 'baseline_psl':
        kl_loss = torch.tensor(0.0, device=device)
        energy_loss = lambda_energy * E_rules_P
        total_loss = ce_loss + energy_loss
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'energy_loss': energy_loss,
        }
        
    if mode == 'baseline_reasoning' or mode == 'TRER':
        
        # 2. KL Divergence (Distillation)
        kl_loss = torch.tensor(0.0, device=device)
        if Q_T is not None and P is not None:
            kl_loss = kl_div(torch.log(P + 1e-12), Q_T.detach())
                
        energy_loss = lambda_energy * E_rules_P
        kl_loss = lambda_distill * kl_loss
        
        total_loss = ce_loss + kl_loss + energy_loss
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'energy_loss': energy_loss,
        }
  

def _backward_pass(loss, model, nn_optimizer, w_rules_optimizer, scaler, max_grad_norm, pbar):
    """
    Handle backward pass with gradient clipping and dual optimizer updates.
    
    Args:
        loss: Total loss tensor
        model: The model
        nn_optimizer: Neural network optimizer
        w_rules_optimizer: Rule weights optimizer
        scaler: GradScaler for AMP
        max_grad_norm: Max gradient norm for clipping
        pbar: Progress bar (for logging step number)
        
    Returns:
        tuple: (nn_grad_norm, w_rules_grad_norm)
    """
    if scaler is not None:
        scaler.scale(loss).backward()

        # 1. Unscale BOTH optimizers explicitly to allow gradient clipping on real magnitudes
        scaler.unscale_(nn_optimizer)
        if w_rules_optimizer is not None:
            scaler.unscale_(w_rules_optimizer)
    else:
        loss.backward()
    
    # 2. Gradient Clipping (now acting on unscaled gradients)
    # Compute norms for logging
    nn_grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for group in nn_optimizer.param_groups for p in group['params']], 
        max_grad_norm
    )
    
    if w_rules_optimizer is not None:
        w_rules_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in w_rules_optimizer.param_groups for p in group['params']], 
            max_grad_norm
        )
    else:
        w_rules_grad_norm = 0.0
    
    # 3. Optimizer Steps
    if scaler is not None:
        # Scaler detects if unscale_() has already been called and doesn't repeat it
        scaler.step(nn_optimizer)

        # Only step w_rules_optimizer if it has gradients (to avoid GradScaler error during warmup)
        if w_rules_optimizer is not None and w_rules_grad_norm > 0.0:
            scaler.step(w_rules_optimizer)
        elif w_rules_optimizer is not None:
            # During warmup, gradients are None, so we must NOT step scaler
            pass

        # 4. Update Scaler
        scaler.update()
    else:
        nn_optimizer.step()
        if w_rules_optimizer is not None and w_rules_grad_norm > 0.0:
            w_rules_optimizer.step()
    
    return nn_grad_norm, w_rules_grad_norm



def train_model(
    mode,
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    save_path,
    lambda_distill=0.0,
    lambda_energy=0.0,
    use_amp=True,
    max_grad_norm=1.0,
    resume_from=None,
    class_weights=None,
    vague_label_id=None,
    is_MATRES=False, # For evaluation flag
    is_I2B2=False,    # For evaluation flag
    is_TBD=False,     # For evaluation flag
    is_TDD=False,     # For evaluation flag
    warmup_epochs=0, # [ADDED]
):

    device = _setup_device(model)
    
    print(f"Training Configuration: epochs={epochs}, lr={lr}, lambda_distill={lambda_distill}, lambda_energy={lambda_energy}, use_amp={use_amp}, max_grad_norm={max_grad_norm}, mode={mode}")
    
    nn_optimizer, w_rules_optimizer = _get_optimizer(model, lr)
    
    # Loss Function
    if class_weights is not None:
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        if mode in ('baseline', 'baseline_psl'):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
        else:
            loss_fn = nn.NLLLoss(weight=class_weights_t)
        print(f"Using Weighted Loss: {class_weights}")
    else:
        if mode in ('baseline', 'baseline_psl'):
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.NLLLoss()

    kl_div = nn.KLDivLoss(reduction='batchmean')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    start_epoch, best_val_f1 = _load_checkpoint(model, nn_optimizer, w_rules_optimizer, resume_from, device)
    
    # Force LR update for both optimizers
    for param_group in nn_optimizer.param_groups:
        param_group['lr'] = lr
    
    if w_rules_optimizer is not None:
        for param_group in w_rules_optimizer.param_groups:
            param_group['lr'] = lr  # Same LR (was lr * 5.0)
    
    print(f"Enforced learning rate: {lr}")

    total_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        model.train()
    
        # Accumulation variables for stats
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_energy_loss = 0.0
        total_energy_P = 0.0
        total_energy_Q = 0.0
        total_energy_rules_P = 0.0
        total_energy_rules_Q = 0.0 
        total_violations_P = 0
        total_violations_Q = 0
        total_groundings = 0
        total_steps = 0
        epoch_rule_stats = defaultdict(lambda: {'weight': 0.0, 'grounding_count': 0, 'vp': 0, 'vq': 0, 'frequency': 0})
        
        # Accumulate metrics for epoch summary
        epoch_losses = []
        epoch_ce_losses = []
        epoch_kl_losses = []
        epoch_energy_losses = []
        epoch_neural_grads = []
        epoch_wrules_grads = []
        epoch_rules_details_P = []
        epoch_rules_details_Q = []
        
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            nn_optimizer.zero_grad()
            if w_rules_optimizer is not None:
                w_rules_optimizer.zero_grad()
                 
            # Move inputs to device
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            e1_marks       = batch["e1_marks"].to(device, non_blocking=True)
            e2_marks       = batch["e2_marks"].to(device, non_blocking=True)
            word_marks     = batch["word_marks"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)
            
            # Optional inputs for Reasoning
            e1_type_ids = batch.get("e1_type_ids")
            if e1_type_ids is not None: e1_type_ids = e1_type_ids.to(device, non_blocking=True)
            
            e2_type_ids = batch.get("e2_type_ids")
            if e2_type_ids is not None: e2_type_ids = e2_type_ids.to(device, non_blocking=True)
            
            doc_ids = batch.get("doc_ids", [])
            if isinstance(doc_ids, torch.Tensor):
                doc_ids = doc_ids.tolist()
                
            e1_ids = batch.get('e1_ids', [])
            e2_ids = batch.get('e2_ids', [])
            entity_pairs = list(zip(e1_ids, e2_ids)) if e1_ids and e2_ids else None

            model_args = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'word_marks': word_marks,
                        'e1_marks': e1_marks,
                        'e2_marks': e2_marks,
                        'entity_pairs': entity_pairs,
                        'doc_ids': doc_ids,
                        'e1_type_ids': e1_type_ids,
                        'e2_type_ids': e2_type_ids
                    }
            
            
            
            # Forward Pass
            if use_amp:
                with torch.cuda.amp.autocast(enabled=use_amp): # Fixed: enabled=use_amp
                    # Prepare common args
                    

                    out = model(**model_args)

                    # Extract Logits and Extra Info
                    extracted = _extract_reasoning_stats(out)
                    L_F = extracted['L_F']
                    P_F = extracted['P_F']
                    Q_T = extracted['Q_T']
                    L_nn = extracted['L_nn']
                    P = extracted['P']
                    
                    E_re_P = extracted['E_re_P']
                    E_re_Q = extracted['E_re_Q']
                    E_rules_P = extracted['E_rules_P']
                    E_rules_Q = extracted['E_rules_Q']
                    VC_P = extracted['VC_P']
                    VC_Q = extracted['VC_Q']
                    GR = extracted['GR']
                    rule_details_Q = extracted['rule_details_Q']
                    rule_details_P = extracted['rule_details_P']
                    T = extracted['T']
                    
                    # Compute all losses
                    loss_result = _compute_losses(labels, L_F, Q_T, P, E_rules_P, loss_fn, kl_div, lambda_distill, lambda_energy, vague_label_id, device, mode)
                    
                    loss = loss_result['total_loss']
                    ce_loss = loss_result['ce_loss']
                    kl_loss = loss_result['kl_loss']
                    energy_loss = loss_result['energy_loss']
                    
                    # Helper to safely extract value
                    def _to_float(x):
                        return x.item() if torch.is_tensor(x) else float(x)

                # Backward pass with gradient clipping and optimizer steps
                if loss.requires_grad:
                    nn_grad_norm, w_rules_grad_norm = _backward_pass(
                        loss, model, nn_optimizer, w_rules_optimizer, scaler, max_grad_norm, pbar
                    )
                else:
                    nn_grad_norm, w_rules_grad_norm = 0.0, 0.0
                
            else:
                
                out = model(**model_args)
             
                # Extract Logits and Extra Info
                extracted = _extract_reasoning_stats(out)
                L_F = extracted['L_F']
                P_F = extracted['P_F']
                Q_T = extracted['Q_T']
                L_nn = extracted['L_nn']
                P = extracted['P']
                
                E_re_P = extracted['E_re_P']
                E_re_Q = extracted['E_re_Q']
                E_rules_P = extracted['E_rules_P']
                E_rules_Q = extracted['E_rules_Q']
                VC_P = extracted['VC_P']
                VC_Q = extracted['VC_Q']
                GR = extracted['GR']
                rule_details_Q = extracted['rule_details_Q']
                rule_details_P = extracted['rule_details_P']
                T = extracted['T']
                # Compute all losses
                loss_result = _compute_losses(labels, L_F, Q_T, P, E_rules_P, loss_fn, kl_div, lambda_distill, lambda_energy, vague_label_id, device, mode)
                
                loss = loss_result['total_loss']
                ce_loss = loss_result['ce_loss']
                kl_loss = loss_result['kl_loss']
                energy_loss = loss_result['energy_loss']

                # Backward pass
                if loss.requires_grad:
                    nn_grad_norm, w_rules_grad_norm = _backward_pass(
                        loss, model, nn_optimizer, w_rules_optimizer, None, max_grad_norm, pbar
                    )
                else:
                    nn_grad_norm, w_rules_grad_norm = 0.0, 0.0
                

                
            # Tracking
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kl += kl_loss.item()
            total_energy_loss += energy_loss
            
            total_energy_P += float(E_re_P)
            total_energy_Q += float(E_re_Q)
            total_energy_rules_P += float(E_rules_P)
            total_energy_rules_Q += float(E_rules_Q)
            total_violations_P += VC_P
            total_violations_Q += VC_Q
            total_groundings += GR
            total_steps += T
                
            # Accumulate metrics for epoch summary
            epoch_losses.append(_to_float(loss))
            epoch_ce_losses.append(_to_float(ce_loss))
            epoch_kl_losses.append(_to_float(kl_loss))
            epoch_energy_losses.append(_to_float(energy_loss))
            
            epoch_rules_details_P.append(rule_details_P)
            epoch_rules_details_Q.append(rule_details_Q)

            # Aggregate per-rule stats
            if isinstance(rule_details_Q, dict):
                for r_name, r_info in rule_details_Q.items():
                    if not isinstance(r_info, dict):
                        continue
                    epoch_rule_stats[r_name]['weight'] += float(r_info.get('weight', 0.0))
                    epoch_rule_stats[r_name]['grounding_count'] += int(r_info.get('grounding_count', 0))
                    epoch_rule_stats[r_name]['vq'] += int(r_info.get('violation_count', 0))
                    epoch_rule_stats[r_name]['frequency'] += 1

            if isinstance(rule_details_P, dict):
                for r_name, r_info in rule_details_P.items():
                    if not isinstance(r_info, dict):
                        continue
                    epoch_rule_stats[r_name]['vp'] += int(r_info.get('violation_count', 0))
        

            
            # Extract gradient info for accumulation
            neural_grad_norm = 0.0
            rule_grad_mean = 0.0
            if mode == 'baseline_reasoning' or mode == 'TRER':
                if hasattr(model, 'reasoning') and hasattr(model.reasoning, 'w_rules_logits') and model.reasoning.w_rules_logits.grad is not None:
                    rule_grad_mean = model.reasoning.w_rules_logits.grad.mean().item()
                for p in model.neural.parameters():
                    if p.grad is not None:
                        neural_grad_norm += p.grad.norm().item()
            elif mode == 'baseline_psl':
                if hasattr(model, 'psl_layer') and hasattr(model.psl_layer, 'w_rules_logits') and model.psl_layer.w_rules_logits.grad is not None:
                    rule_grad_mean = model.psl_layer.w_rules_logits.grad.mean().item()
                for p in model.neural.parameters():
                    if p.grad is not None:
                        neural_grad_norm += p.grad.norm().item()
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        neural_grad_norm += p.grad.norm().item()
                        
            epoch_neural_grads.append(neural_grad_norm)
            epoch_wrules_grads.append(rule_grad_mean)
            
            
            update_freq = 1 if len(train_loader) < 500 or (hasattr(train_loader.batch_sampler, 'batch_size') and train_loader.batch_sampler.batch_size == 1) else 10
            
            if epoch % 1 == 0 and pbar.n % update_freq == 0:
                
                 metrics = {'Loss': f"{loss.item():.4f}"}
                 if w_rules_optimizer is not None:
                        metrics.update({
                            'L_CE': f"{ce_loss.item():.4f}",
                            'L_KL': f"{kl_loss.item():.4f}",
                            'L_EP': f"{energy_loss:.4f}",
                            'E_rule_P': f"{E_rules_P:.4f}",
                            'E_rule_Q': f"{E_rules_Q:.4f}",
                            'VP': f"{VC_P}",
                            'VQ': f"{VC_Q}",
                            # 'GR': f"{GR}",
                            'T': f"{T}"
                        })
                 pbar.set_postfix(metrics)

        # --- EVALUATION ---
        cm, val_f1, p, r, _, _, _ = evaluate(
            val_loader, 
            model, 
            device=device,
            is_MATRES=is_MATRES,
            is_I2B2=is_I2B2,
            is_TBD=is_TBD,
            is_TDD=is_TDD,
            vague_label_id=vague_label_id
        )
       
       
        # Print Epoch Metrics Summary Table
        def _get_stats(vals):
            if not vals: return 0.0, 0.0, 0.0, 0.0
            arr = np.array(vals, dtype=np.float32)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0: return 0.0, 0.0, 0.0, 0.0
            return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

        if len(epoch_ce_losses) > 0:
            print(f"\n[EPOCH {epoch} METRICS SUMMARY]")
            print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
            print("-" * 65)
            
            m, s, mn, mx = _get_stats(epoch_ce_losses)
            print(f"{'L_CE':<15} {m:<12.6f} {s:<12.6f} {mn:<12.6f} {mx:<12.6f}")
            
            if w_rules_optimizer is not None:
                m, s, mn, mx = _get_stats(epoch_kl_losses)
                print(f"{'L_KL':<15} {m:<12.6f} {s:<12.6f} {mn:<12.6f} {mx:<12.6f}")
                m, s, mn, mx = _get_stats(epoch_energy_losses)
                print(f"{'L_E_rule':<15} {m:<12.6f} {s:<12.6f} {mn:<12.6f} {mx:<12.6f}")
            
            m, s, mn, mx = _get_stats(epoch_neural_grads)
            print(f"{'Neural_Grad':<15} {m:<12.6e} {s:<12.6e} {mn:<12.6e} {mx:<12.6e}")
            
            if w_rules_optimizer is not None:
                m, s, mn, mx = _get_stats(epoch_wrules_grads)
                print(f"{'W_Rules_Grad':<15} {m:<12.6e} {s:<12.6e} {mn:<12.6e} {mx:<12.6e}")
            print()
        
        # [MODIFIED] Streamlined Epoch Print
        log_str = f"Epoch {epoch}: Loss={total_loss:.4f} | Val F1={val_f1:.4f} P={p:.4f} R={r:.4f}"
        if w_rules_optimizer is not None:
            log_str += f" | E_rule_P={total_energy_rules_P:.2e} E_rule_Q={total_energy_rules_Q:.2e} | Viol_P={int(total_violations_P)} Viol_Q={int(total_violations_Q)} GR={int(total_groundings)} T={int(T)}"
        
        print(log_str)
        
        # Print Rule Stats
        if len(epoch_rule_stats) > 0:
            print("\n  [Rule Statistics]")
        for r_name in sorted(epoch_rule_stats.keys()):
            info = epoch_rule_stats[r_name]
            freq = max(1, info['frequency'])
            avg_weight = info['weight'] / freq
            print(f"  - {r_name}: Weight={avg_weight:.4f} | GR_Count={int(info['grounding_count'])} | VP={int(info.get('vp', 0))} | VQ={int(info.get('vq', 0))}")
        print("")
        
        best_val_f1 = _save_checkpoint(model, nn_optimizer, w_rules_optimizer, epoch, val_f1, best_val_f1, save_path)
    
    print(f"Training finished in {time.time() - total_start_time:.2f}s.")