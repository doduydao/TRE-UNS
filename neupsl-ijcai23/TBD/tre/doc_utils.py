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

# from i2b2_eval import build_relations_from_df, evaluate_tempeval3



# ---------- Helper: chuẩn hóa attention ----------

def normalize_attention(alpha, word_mask=None, eps: float = 1e-8):
    """
    alpha: [B,W]
    word_mask: [B,W] bool hoặc None

    Trả về: alpha_norm [B,W] đã chuẩn hóa theo word hợp lệ.
    """
    if word_mask is not None:
        mask = word_mask.to(alpha.dtype)  # [B,W]
        alpha = alpha * mask

    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(eps)
    return alpha


# ---------- 1) CE chính ----------

def loss_ce(logits, labels):
    """
    CrossEntropy chính.
    """
    return F.cross_entropy(logits, labels)


# ---------- 2) Regularize Z (entropy + diversity) ----------

def loss_Z_regularization(alpha_P, alpha_Z, word_mask=None):
    """
    Tính 2 loss:
      - Entropy của alpha_Z (muốn attention sắc, entropy thấp)
      - Diversity P–Z (cos_sim cao -> phạt, muốn Z khác P)

    alpha_P, alpha_Z: [B,W]
    word_mask: [B,W] bool hoặc None

    Trả về:
      loss_Z_ent, loss_Z_div  (chưa nhân lambda)
    """
    device = alpha_P.device
    dtype  = alpha_P.dtype

    alpha_P = normalize_attention(alpha_P, word_mask)
    alpha_Z = normalize_attention(alpha_Z, word_mask)

    # 2.1) Entropy penalty cho Z
    eps = 1e-8
    entropy_Z = -(alpha_Z.clamp_min(eps) * alpha_Z.clamp_min(eps).log()).sum(dim=-1).mean()
    loss_Z_ent = entropy_Z.to(device=device, dtype=dtype)

    # 2.2) Diversity P–Z: cos_sim cao -> phạt
    sim_PZ = F.cosine_similarity(alpha_P, alpha_Z, dim=-1).mean()
    loss_Z_div = sim_PZ.to(device=device, dtype=dtype)

    return loss_Z_ent, loss_Z_div


# ---------- 3) PSL document-level (transitivity + symmetry) ----------

def loss_psl_document(
    probs,
    e1_ids,
    e2_ids,
    doc_ids,
    rel_index,
):
    """
    PSL document-level:

    - Transitivity Dependencies:
        BBB, BOB, OBB, OOO, AAA, AOA, OAA
    - Symmetry Dependencies:
        BA, AB, OO

    probs:  [B,C]  (softmax(logits_main))
    e1_ids, e2_ids, doc_ids: list độ dài B (string hoặc int)
    rel_index: dict {"BEFORE": idx_before, "AFTER": idx_after, "OVERLAP": idx_overlap}

    Trả về:
      loss_PSL_trans, loss_PSL_sym  (chưa nhân lambda)
    """
    device = probs.device
    dtype  = probs.dtype

    idx_before  = rel_index["BEFORE"]
    idx_after   = rel_index["AFTER"]
    idx_overlap = rel_index["OVERLAP"]

    from collections import defaultdict as dd
    doc_to_indices = dd(list)
    for i, d in enumerate(doc_ids):
        doc_to_indices[d].append(i)

    psl_trans_total = torch.zeros((), device=device, dtype=dtype)
    psl_sym_total   = torch.zeros((), device=device, dtype=dtype)
    n_trans = 0
    n_sym   = 0

    # Lặp từng doc
    for d, idxs in doc_to_indices.items():
        if len(idxs) < 2:
            continue

        # mapping (e1,e2) -> batch_idx
        pair_map = {}
        for bi in idxs:
            key = (e1_ids[bi], e2_ids[bi])
            pair_map[key] = bi

        # ===== 1) Transitivity Dependencies =====
        # (A,B) = i_idx, (B,C) = j_idx, (A,C) = k_idx
        for i_idx in idxs:
            a = e1_ids[i_idx]
            b = e2_ids[i_idx]

            for j_idx in idxs:
                if i_idx == j_idx:
                    continue
                if e1_ids[j_idx] != b:
                    continue  # phải cùng B

                c = e2_ids[j_idx]
                k_idx = pair_map.get((a, c), None)  # (A,C)
                if k_idx is None:
                    continue

                # Xác suất quan hệ
                p_ab_B = probs[i_idx, idx_before]
                p_ab_A = probs[i_idx, idx_after]
                p_ab_O = probs[i_idx, idx_overlap]

                p_bc_B = probs[j_idx, idx_before]
                p_bc_A = probs[j_idx, idx_after]
                p_bc_O = probs[j_idx, idx_overlap]

                p_ac_B = probs[k_idx, idx_before]
                p_ac_A = probs[k_idx, idx_after]
                p_ac_O = probs[k_idx, idx_overlap]

                # --- BBB: Before(A,B) ∧ Before(B,C) → Before(A,C)
                viol_BBB = F.relu(p_ab_B + p_bc_B - p_ac_B - 1.0)

                # --- BOB: Before(A,B) ∧ Overlap(B,C) → Before(A,C)
                viol_BOB = F.relu(p_ab_B + p_bc_O - p_ac_B - 1.0)

                # --- OBB: Overlap(A,B) ∧ Before(B,C) → Before(A,C)
                viol_OBB = F.relu(p_ab_O + p_bc_B - p_ac_B - 1.0)

                # --- OOO: Overlap(A,B) ∧ Overlap(B,C) → Overlap(A,C)
                viol_OOO = F.relu(p_ab_O + p_bc_O - p_ac_O - 1.0)

                # --- AAA: After(A,B) ∧ After(B,C) → After(A,C)
                viol_AAA = F.relu(p_ab_A + p_bc_A - p_ac_A - 1.0)

                # --- AOA: After(A,B) ∧ Overlap(B,C) → After(A,C)
                viol_AOA = F.relu(p_ab_A + p_bc_O - p_ac_A - 1.0)

                # --- OAA: Overlap(A,B) ∧ After(B,C) → After(A,C)
                viol_OAA = F.relu(p_ab_O + p_bc_A - p_ac_A - 1.0)

                # Cộng dồn tất cả luật transitivity
                psl_trans_total = (
                    psl_trans_total
                    + viol_BBB + viol_BOB + viol_OBB
                    + viol_OOO + viol_AAA + viol_AOA + viol_OAA
                )
                # mỗi triple (A,B,C) kích hoạt 7 luật
                n_trans += 7

        # ===== 2) Symmetry Dependencies =====
        # (A,B) = i_idx, (B,A) = j_idx
        for i_idx in idxs:
            a = e1_ids[i_idx]
            b = e2_ids[i_idx]
            j_idx = pair_map.get((b, a), None)  # (B,A)
            if j_idx is None:
                continue

            p_ab_B = probs[i_idx, idx_before]
            p_ab_A = probs[i_idx, idx_after]
            p_ab_O = probs[i_idx, idx_overlap]

            p_ba_B = probs[j_idx, idx_before]
            p_ba_A = probs[j_idx, idx_after]
            p_ba_O = probs[j_idx, idx_overlap]

            # --- BA: Before(A,B) → After(B,A)
            viol_BA = F.relu(p_ab_B - p_ba_A)

            # --- AB: After(A,B) → Before(B,A)
            viol_AB = F.relu(p_ab_A - p_ba_B)

            # --- OO: Overlap(A,B) → Overlap(B,A)
            viol_OO = F.relu(p_ab_O - p_ba_O)

            psl_sym_total = psl_sym_total + viol_BA + viol_AB + viol_OO
            n_sym += 3   # 3 luật cho mỗi cặp (A,B)/(B,A)

    return psl_trans_total, psl_sym_total


# ---------- 4) Orchestrator: tổng hợp loss ----------


def evaluate(dataloader, model, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)      # [T,L_pad]
            attention_mask = batch["attention_mask"].to(device)
            word_indices   = batch["word_indices"]
            e1_indices     = batch["e1_indices"]
            e2_indices     = batch["e2_indices"]
            labels         = batch["labels"].to(device)
            L_pad          = batch["L_pad"]
            T              = batch["T"]

            if labels.numel() == 0:
                continue

            logits_main, logits_Z, extras = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_indices=word_indices,
                e1_indices=e1_indices,
                e2_indices=e2_indices,
                L_pad=L_pad,
                T=T,
            )

            preds = torch.argmax(logits_main, dim=1)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(preds.detach().cpu().numpy().tolist())

    if len(all_labels) == 0:
        # tránh crash nếu dev set toàn doc không có pair
        cm  = None
        f1  = 0.0
        p   = 0.0
        r   = 0.0
        precisions = recalls = f1s = []
        return (cm, f1, p, r, precisions, recalls, f1s)

    acc = accuracy_score(all_labels, all_predictions)
    f1  = f1_score(all_labels, all_predictions, average='weighted')
    p   = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    r   = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    cm  = confusion_matrix(all_labels, all_predictions)
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )

    metrics = (cm, f1, p, r, precisions, recalls, f1s)
    return metrics
