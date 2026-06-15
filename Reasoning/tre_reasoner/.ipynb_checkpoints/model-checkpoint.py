import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning import (
    UnifiedNeuralReasoningLayer,
    load_rule_file,
    create_default_predicate_registry,
)
from reasoning_v2 import (
    UnifiedNeuralReasoningLayerV2,
    parse_rule_line_v2,
)

# ===================== Pooling helpers =====================
def pool_entity_emb(X, marks):
    """Pool entity embedding trên token-level: X=[B,L,H], marks=[B,L] or [B,1,L]."""
    if marks.dim() == 2:
        marks = marks.unsqueeze(1)
    elif marks.dim() == 3 and marks.size(1) != 1:
        raise ValueError(f"marks phải [B,L] hoặc [B,1,L], hiện tại={marks.shape}")

    marks = marks.float()                    # [B,1,L]
    mask_sum = marks.sum(dim=2, keepdim=True).clamp(min=1e-6)
    embs = torch.bmm(marks, X) / mask_sum    # [B,1,H]
    return embs.squeeze(1)                   # [B,H]


def pool_word_emb(X, marks):
    """Pool word embeddings: X=[B,L,H], marks=[B,W,L]."""
    marks = marks.float()
    mask_sum = marks.sum(dim=2, keepdim=True)          # [B,W,1]
    word_mask = (mask_sum.squeeze(-1) > 0)             # [B,W]
    mask_sum_safe = mask_sum.clamp(min=1e-6)

    word_embs = torch.bmm(marks, X) / mask_sum_safe    # [B,W,H]
    word_embs = word_embs * word_mask.unsqueeze(-1)
    return word_embs, word_mask


# ===================== LatentReasoning block =====================

class TextEncoder(nn.Module):
    """Self-attention block: Q=K=V=[P,W]."""
    def __init__(self, d_model, nhead=8, dim_ff=2048, p_drop=0.1):
        super().__init__()
        self.ln_q  = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mha   = nn.MultiheadAttention(
            d_model, nhead, dropout=p_drop, batch_first=True
        )
        self.drop_attn = nn.Dropout(p_drop)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(p_drop)
        )

    def forward(self, Q, key_padding_mask=None, need_weights=False):
        Mn = self.ln_kv(Q)
        Qn = self.ln_q(Q)
        out, attn = self.mha(
            query=Qn,
            key=Mn,
            value=Mn,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        Q = Q + self.drop_attn(out)
        Q = Q + self.ff(self.ln_ff(Q))
        return Q, (attn if need_weights else None)

class TREPotential(nn.Module):
    def __init__(self,
                 bert,
                 num_classes,
                 enc_layers=2,
                 nhead=8,
                 p_drop=0.1,
                 freeze_bert=False):
        super().__init__()
        self.bert = bert
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.H = bert.config.hidden_size
        self.num_classes = num_classes

        # e1,e2 → P0
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(2 * self.H),
            nn.Linear(2 * self.H, self.H),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        # text encoder
        self.joint_layers = nn.ModuleList([
            TextEncoder(self.H, nhead=nhead, dim_ff=4*self.H, p_drop=p_drop)
            for _ in range(enc_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(self.H, num_classes)
        )

    def _pool_entities_and_words(self, X_tok, word_marks, e1_marks, e2_marks):
        word_embs, word_mask = pool_word_emb(X_tok, word_marks)
        e1 = pool_entity_emb(X_tok, e1_marks)
        e2 = pool_entity_emb(X_tok, e2_marks)
        return e1, e2, word_embs, word_mask

    def _encode(self, input_ids, attention_mask, word_marks, e1_marks, e2_marks):
        
        X_tok = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        e1, e2, W, word_mask = self._pool_entities_and_words(
            X_tok, word_marks, e1_marks, e2_marks
        )

        P = self.pair_proj(torch.cat([e1, e2], dim=-1)).unsqueeze(1) #(B, 1, H)

        word_padding = ~word_mask

        B, W_len, H = W.shape
        # W: (B, W_len, H)
        Q = torch.cat([P, W], dim=1)
    
        # key_padding_mask tương ứng:
        # P -> 1 vị trí, không bị mask
        # W -> dùng mask gốc
        kpm = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=W.device),  # cho P
            word_padding                                     # cho W
        ], dim=1)
    
        P_att = None  # attention của P đến W
   
        for i, layer in enumerate(self.joint_layers):
            need_w = (i == len(self.joint_layers)-1)
            Q, attn = layer(Q, key_padding_mask=kpm, need_weights=need_w)
    
            if need_w and attn is not None:
                # attn shape: (B, num_heads, Q_len, Q_len)
                # P là index 0, W bắt đầu từ index 1
                P_att = attn[:, 0, 1:]  # attention từ P tới toàn bộ W
    
        P_out =  Q[:, 0:1]
        W_out =  Q[:, 1:]
        return P_out, W_out, P_att, word_mask

    
    # -------------------------------------
    def forward(self, input_ids, attention_mask, word_marks, e1_marks, e2_marks):
        enc, W, P_att, word_mask = self._encode(input_ids, attention_mask, word_marks, e1_marks, e2_marks)
        L = self.classifier(enc.squeeze(1))
        P = F.softmax(L, dim=-1)
        return {"L_F": L, "P_F": P, "Q_T": None, "L_nn": L, 'P': P, 'extras': {'P_att': P_att, 'word_mask': word_mask}}
    
    
    @staticmethod
    def extract_topk_words(extras, words_spacy, topk=10):
        """
        Từ extras sau inference, trích top-k word theo P_att

        extras:
          - "P_att": [B,W]
          - "word_mask": [B,W] hoặc None

        return:
          topk_words_P: list[list[(idx, text, score)]]
        """
        P_att = extras["P_att"]      # [B,W]
        word_mask = extras.get("word_mask", None)  # [B,W] or None

        B, W_len = P_att.shape
        topk_words_P = []

        for b in range(B):
            wlist = None
            if isinstance(words_spacy, (list, tuple)) and b < len(words_spacy):
                wlist = words_spacy[b]

            # xác định word hợp lệ
            if word_mask is not None:
                valid_idx = torch.nonzero(word_mask[b], as_tuple=False).flatten().tolist()
            else:
                valid_idx = list(range(W_len))

            pairs_P = []

            for wi in valid_idx:
                score_P = float(P_att[b, wi])

                if wlist is not None and wi < len(wlist):
                    w = wlist[wi]
                    pairs_P.append((wi, w, score_P))
            pairs_P.sort(key=lambda x: x[2], reverse=True)
 
            topk_words_P.append(pairs_P[:topk])
        return topk_words_P


class ReasoningModel(nn.Module):
    """
    Scenario 3: Reasoning (Neuro-Symbolic).
    Neural + Reasoning Layer (used in Forward pass).
    """
    def __init__(self, neural, reasoning_layer, num_labels=None, output_option='direct_q'):
        super().__init__()
        self.neural = neural
        self.reasoning = reasoning_layer
        self.output_option = output_option
        self.num_labels = num_labels
        
        if output_option == 'refinement':
            self.refinement_alpha = nn.Parameter(torch.ones(1, num_labels)) # Diagonal option

    
    def forward(
        self, 
        input_ids,
        attention_mask,
        word_marks,
        e1_marks,
        e2_marks,
        entity_pairs=None, 
        doc_ids=None, 
        e1_type_ids=None, 
        e2_type_ids=None,

    ):
        # 1. Base Logits from neural (P_0)
        nn = self.neural(input_ids, attention_mask, word_marks, e1_marks, e2_marks)
        L_nn = nn['L_nn']
        P = nn['P']
        extras = nn['extras']
        
        # 3. Reasoning to obtain Q (logic-driven distribution)
        Q_T = None 
        L_F = None
        P_F = None
        
        if entity_pairs is not None:
            Q_T, energy_history = self.reasoning(
                P=P,
                entity_pairs=entity_pairs,
                doc_ids=doc_ids,
                e1_types=e1_type_ids,
                e2_types=e2_type_ids
            )
            
            # Forward energy_history - extraction happens in train.py
            extras['energy_history'] = energy_history
    
        # 4. Refinement logic based on output_option
        if self.output_option == 'direct_q':
                L_F = torch.log(Q_T + 1e-6)
                P_F = Q_T
                
        if self.output_option == 'refinement':
                L_F = L_nn + self.refinement_alpha * torch.log(Q_T + 1e-6)
                P_F = torch.softmax(L_F, dim=1)
        
        return {"L_F": L_F, "P_F": P_F, "Q_T": Q_T, "L_nn": L_nn, 'P': P, 'extras': extras}


class BaseLine(nn.Module):
    def __init__(self,
                 bert,
                 num_classes,
                 enc_layers=2,
                 nhead=8,
                 p_drop=0.1,
                 freeze_bert=False):
        super().__init__()
        self.bert = bert
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.H = bert.config.hidden_size
        self.num_classes = num_classes

        # e1,e2 → P0
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(2 * self.H),
            nn.Linear(2 * self.H, self.H),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.H),
            nn.Linear(self.H, self.H),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(self.H, num_classes)
        )

    def _encode(self, input_ids, attention_mask, word_marks, e1_marks, e2_marks):
        X_tok = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        e1 = pool_entity_emb(X_tok, e1_marks)
        e2 = pool_entity_emb(X_tok, e2_marks)
        P = self.pair_proj(torch.cat([e1, e2], dim=-1))
        return P

    
    # -------------------------------------
    def forward(self,
                input_ids,
                attention_mask,
                word_marks,
                e1_marks,
                e2_marks,
                entity_pairs=None, 
                doc_ids=None, 
                e1_type_ids=None, 
                e2_type_ids=None
               ):
        enc = self._encode(input_ids, attention_mask, word_marks, e1_marks, e2_marks)
        L = self.classifier(enc)
        P = F.softmax(L, dim=-1)
        
        return {"L_F": L, "P_F": P, "Q_T": None, "L_nn": L, 'P': P, 'extras': {}}


class BaseLinePSL(nn.Module):
    """
    Baseline classifier + PSL regularization helper.
    Prediction path is baseline-only; PSL is used only for loss regularization.
    """
    def __init__(self, baseline_model, psl_layer):
        super().__init__()
        self.neural = baseline_model
        self.psl_layer = psl_layer

    def forward(self,
                input_ids,
                attention_mask,
                word_marks,
                e1_marks,
                e2_marks,
                entity_pairs=None,
                doc_ids=None,
                e1_type_ids=None,
                e2_type_ids=None):
        out = self.neural(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_marks=word_marks,
            e1_marks=e1_marks,
            e2_marks=e2_marks,
        )

        extras = out.get('extras', {})
        P = out['P']

        if entity_pairs is not None and doc_ids is not None:
            ctx = self.psl_layer.build_context(P, entity_pairs, doc_ids, e1_type_ids, e2_type_ids)
            psl = self.psl_layer._compute_rule_energy(P, ctx)
            extras['psl_stats'] = {
                'E_rules': psl.get('E_rules', 0.0),
                'GR': psl.get('total_groundings', 0.0),
                'VC': psl.get('violation_count', 0.0),
                'rule_details': psl.get('rule_details', {}),
            }
        else:
            extras['psl_stats'] = {
                'E_rules': 0.0,
                'GR': 0.0,
                'VC': 0.0,
                'rule_details': {},
            }

        out['extras'] = extras
        return out
    

# ============================================================
# 3. FACTORY: create_model
# ============================================================

def create_model(
    mode='baseline',
    bert=None,
    num_classes=3,
    num_types=0,
    type_map=None,
    relation_map=None,
    enc_layers=2,
    nhead=8,
    p_drop=0.1,
    step_size=1.0,       # [MODIFIED] Increased from 0.1
    lambda_kl: float = 0.001,  # [MODIFIED] Allow even more freedom for Q
    smooth_tau: float = 0.1,
    lambda_entropy: float = 0.0,
    max_steps: int = 50,
    tol: float = 1e-6,      # [ADDED] Stricter tolerance
    rule_file_path=None,
    freeze_bert=False,
    use_deq: bool = False,
    output_option: str = 'direct_q', # [ADDED] 'direct_q' or 'refinement'
    tokenizer = None, # [ADDED] Tokenizer for special ID extraction
    learn_rule_weights: bool = True, # [ADDED]
    initial_rule_weight: float = 0.0, # [ADDED]
    rule_chunk_size: int = 1,
):
    if mode=='baseline': 
        return BaseLine(bert, num_classes, freeze_bert=freeze_bert)

    if mode=='baseline_psl':
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        rule_templates = [parse_rule_line_v2(line) for line in lines if parse_rule_line_v2(line)]

        nn = BaseLine(bert, num_classes, freeze_bert=freeze_bert)
        psl_layer = UnifiedNeuralReasoningLayerV2(
            rule_templates=rule_templates,
            num_relations=num_classes,
            num_types=num_types,
            step_size=step_size,
            lambda_kl=lambda_kl,
            lambda_entropy=lambda_entropy,
            smooth_tau=smooth_tau,
            max_steps=max_steps,
            tol=tol,
            type_name_to_id=type_map,
            relation_to_index=relation_map,
            use_deq=use_deq,
            learn_rule_weights=learn_rule_weights,
            rule_chunk_size=rule_chunk_size,
        )
        return BaseLinePSL(nn, psl_layer)
        
    if mode=='baseline_reasoning':
        # 1. Load rules
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        rule_templates = [parse_rule_line_v2(line) for line in lines if parse_rule_line_v2(line)]
    
        # 2. Encoder
        # Use Baseline encoder (BERT + MLP) as requested
        nn = BaseLine(bert, num_classes, freeze_bert=freeze_bert)
        
        # 3. Reasoner
        reasoning_layer = UnifiedNeuralReasoningLayerV2(
            rule_templates=rule_templates,
            num_relations=num_classes,
            num_types=num_types,
            step_size=step_size,
            lambda_kl=lambda_kl,
            lambda_entropy=lambda_entropy,
            smooth_tau=smooth_tau,
            max_steps=max_steps,
            tol=tol,
            type_name_to_id=type_map,
            relation_to_index=relation_map,
            use_deq=use_deq, # Pass use_deq
            learn_rule_weights=learn_rule_weights,
            rule_chunk_size=rule_chunk_size,
        )
        
        return ReasoningModel(nn, reasoning_layer, num_labels=num_classes, output_option=output_option)

    if mode=='TRER':
        with open(rule_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        rule_templates = [parse_rule_line_v2(line) for line in lines if parse_rule_line_v2(line)]
    
        # 2. Encoder
        nn = TREPotential(bert,
                     num_classes,
                     enc_layers=enc_layers,
                     nhead=nhead,
                     p_drop=p_drop,
                     freeze_bert=freeze_bert)
        # 3. Reasoner
        reasoning_layer = UnifiedNeuralReasoningLayerV2(
            rule_templates=rule_templates,
            num_relations=num_classes,
            num_types=num_types,
            step_size=step_size,
            lambda_kl=lambda_kl,
            lambda_entropy=lambda_entropy,
            smooth_tau=smooth_tau,
            max_steps=max_steps,
            tol=tol,
            type_name_to_id=type_map,
            relation_to_index=relation_map,
            use_deq=use_deq, # Pass use_deq
            learn_rule_weights=learn_rule_weights,
            rule_chunk_size=rule_chunk_size,
        )
    
        return ReasoningModel(nn, reasoning_layer, num_labels=num_classes, output_option=output_option)

    return None
    