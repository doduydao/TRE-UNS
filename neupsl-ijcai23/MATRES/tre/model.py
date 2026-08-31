import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ===================== Pooling helpers =====================
def pool_entity_emb(X, marks):
    """Pool entity embeddings at the token level: X=[B,L,H], marks=[B,L] or [B,1,L]."""
    if marks.dim() == 2:
        marks = marks.unsqueeze(1)
    elif marks.dim() == 3 and marks.size(1) != 1:
        raise ValueError(f"marks must be [B,L] or [B,1,L], current={marks.shape}")

    marks = marks.float()                    # [B,1,L]
    mask_sum = marks.sum(dim=2, keepdim=True).clamp(min=1e-6)
    embs = torch.bmm(marks, X) / mask_sum    # [B,1,H]
    return embs.squeeze(1)                   # [B,H]

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
        P = self._encode(input_ids, attention_mask, word_marks, e1_marks, e2_marks)
        L = self.classifier(P)
        return {"LF": L}
