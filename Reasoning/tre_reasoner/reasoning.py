import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import re

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        if torch.rand(1).item() < 0.01: # Print occasionally
            print(f" [REVERSAL] GradOut (from Loss): {grad_output.mean().item():.2e} -> GradIn (to Logits): {grad_input.mean().item():.2e} Alpha={ctx.alpha}")
        return grad_input, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ============================================================
# 1. DSL: Predicate, RuleTemplate, parser rules.txt
# ============================================================

@dataclass
class Predicate:
    """
    Ví dụ:
        name = "Before", args = ["E1", "E2"]
        name = "Type",   args = ["E1", "PROBLEM"]
    """
    name: str
    args: List[str]
    is_negated: bool = False


@dataclass
class RuleTemplate:
    """
    Ví dụ:
        R1: Type(E1, PROBLEM) & Type(E2, TEST) & BeforeType(PROBLEM, TEST) => Before(E1, E2)
        Có thể thêm [HARD] ở RHS để đánh dấu hard rule.
    """
    name: str
    lhs: List[Predicate]
    rhs: Predicate
    is_hard: bool = False


PRED_PATTERN = re.compile(r'\s*([!~]?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*')


def parse_predicate(text: str) -> Predicate:
    """
    Parse chuỗi: "Before(E1, E2)" -> Predicate("Before", ["E1","E2"])
    """
    m = PRED_PATTERN.match(text)
    if not m:
        raise ValueError(f"Cannot parse predicate: {text}")
    neg_char = m.group(1)
    name = m.group(2)
    args_str = m.group(3)
    args = [a.strip() for a in args_str.split(",") if a.strip()]
    
    is_negated = (neg_char in ['!', '~'])
    return Predicate(name=name, args=args, is_negated=is_negated)


def parse_rule_line(line: str) -> Optional[RuleTemplate]:
    """
    Parse line dạng:
        R1: A(E1,E2) & B(E2,E3) => C(E1,E3) [HARD]
    """
    if "#" in line:
        line = line.split("#", 1)[0]
    line = line.strip()
    if not line:
        return None

    if ":" not in line:
        raise ValueError(f"Missing rule name: {line}")
    name_part, rest = line.split(":", 1)
    rule_name = name_part.strip()

    if "=>" not in rest:
        raise ValueError(f"Missing '=>' in rule: {line}")
    lhs_str, rhs_str = rest.split("=>", 1)
    lhs_str = lhs_str.strip()
    rhs_str = rhs_str.strip()

    lhs_parts = [p.strip() for p in lhs_str.split("&") if p.strip()]
    lhs_preds = [parse_predicate(p) for p in lhs_parts]

    # Check for [HARD] tag in rhs
    is_hard = False
    if "[HARD]" in rhs_str:
        is_hard = True
        rhs_str = rhs_str.replace("[HARD]", "").strip()

    rhs_pred = parse_predicate(rhs_str)
    return RuleTemplate(name=rule_name, lhs=lhs_preds, rhs=rhs_pred, is_hard=is_hard)


def load_rule_file(path):
    rules = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#"):
                continue
            rule = parse_rule_line(line)
            if rule is not None:
                rules.append(rule)
    return rules


# ============================================================
# 2. TermSpec / CompiledRule cho vectorized reasoning
# ============================================================

@dataclass
class TermSpec:
    """
    Representation of a term in a vectorized rule.
    source_type: 'M' (Relation) or 'T' (Type)
    tensor_idx:  Index in M (relation id) or T (type id)
    var_indices: List of global variable indices that this term uses.
                 e.g. nếu global vars = [A, B, C], term P(A, C) => [0, 2].
    """
    source_type: str
    tensor_idx: int
    var_indices: List[int]
    is_negated: bool = False


@dataclass
class CompiledRule:
    """
    Rule đã compile cho vectorized execution.
    LHS: list các TermSpec
    RHS: 1 TermSpec
    num_vars: số biến khác nhau trong rule
    mask_indices: không dùng nữa (trong bản smooth HL-MRF), nhưng giữ để tương thích
    """
    rule_idx: int
    name: str
    lhs_terms: List[TermSpec]
    rhs_term: TermSpec
    num_vars: int
    mask_indices: List[int]
    is_hard: bool = False


# ============================================================
# 3. BatchedContext: thông tin cho một batch document
# ============================================================

@dataclass
class BatchedContext:
    """
    Context cho cả batch, tối ưu cho vectorized computation.

    scatter_indices: dùng để scatter Q -> M:
        M[batch_idx, row, col] = Q[global_idx]
    T_mask: [B, N, num_types] (1-hot type)
    entity_mask: [B, N] (1.0 = entity thật, 0.0 = padding)
    """
    scatter_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    global_indices: torch.Tensor
    T_mask: Optional[torch.Tensor]
    entity_mask: torch.Tensor
    pair_mask: torch.Tensor          # [B, N, N] mask of candidate pairs
    batch_size: int
    max_ent: int


# ============================================================
# 4. Predicate Registry (placeholder, nếu muốn gắn thêm eval thủ công)
# ============================================================

class PredicateRegistry:
    def __init__(self):
        self._fns = {}

    def register(self, name: str, fn):
        self._fns[name] = fn

    def eval(self, pred: Predicate, Q: torch.Tensor, ctx, grounding: Dict[str, int]) -> torch.Tensor:
        if pred.name not in self._fns:
            raise KeyError(f"No evaluator registered for predicate '{pred.name}'")
        return self._fns[pred.name](pred, Q, ctx, grounding)


def create_default_predicate_registry() -> PredicateRegistry:
    reg = PredicateRegistry()
    # Có thể đăng ký thêm nếu muốn eval rule theo kiểu legacy.
    return reg


# ============================================================
# 5. Unified Neural Reasoning Layer (Smooth Convex HL-MRF + EG)
# ============================================================

def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0):
    """
    Anderson acceleration for fixed point iteration: x = f(x).
    """
    bsz, d_dim = x0.shape[0], x0.view(x0.shape[0], -1).shape[1]
    x0_flat = x0.view(bsz, -1)
    x = x0_flat
    
    # History buffer
    X, F = [], []
    
    for k in range(max_iter):
        # f(x)
        fx = f(x.view_as(x0)).view_as(x0_flat)
        
        # Residual
        res = fx - x
        norm = torch.norm(res) / (torch.norm(x) + 1e-9)
        if norm < tol:
            return fx.view_as(x0)
        
        X.append(x)
        F.append(fx)
        
        if len(X) > m:
            X.pop(0)
            F.pop(0)
        
        # Anderson update
        n = len(X)
        if n == 1:
            x = fx * beta + x * (1 - beta)
        else:
            # G_i = F_i - X_i
            G = []
            for i in range(n):
                G.append((F[i] - X[i]).view(-1))
            G = torch.stack(G, dim=1) # [Dim, n] (This flattens batch which is wrong for batched solve!)
            
            GtG = torch.matmul(G.t(), G) # [n, n]
            GtG = GtG + lam * torch.eye(n, device=GtG.device)
            
            try:
                # alpha = (GtG)^-1 * 1 / (1^T (GtG)^-1 1)
                # Solve H * alpha = 1
                ones = torch.ones(n, 1, device=GtG.device)
                alpha_unc = torch.linalg.solve(GtG, ones)
                alpha = alpha_unc / alpha_unc.sum()
                alpha = alpha.squeeze() # [n]
                
                # x_{k+1} = \sum alpha_i F_i
                x_new = torch.zeros_like(x)
                for i in range(n):
                    x_new += alpha[i] * F[i]
                x = x_new
            except:
                x = fx * beta + x * (1 - beta)

    return x.view_as(x0)


class ImplicitReasoningFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, layer, entity_pairs, doc_ids, e1_types, e2_types, doc_context=None):
        # 1. Forward: Find Fixed Point Q*
        with torch.no_grad():
            if doc_context is not None:
                c = doc_context
            else:
                c = layer._build_doc_contexts(P, entity_pairs, doc_ids, e1_types, e2_types)
            
            Q = P.clone()
            
            # Define step function for solver
            eps = 1e-12
            def step_fn(CurrQ):
                # Gradient descent step
                with torch.enable_grad():
                    CurrQ = CurrQ.detach().requires_grad_(True)
                    # E = E_rules + lambda_ent * H + lambda_kl * KL
                    energy_result = layer._compute_reason_energy(P, CurrQ, c)
                    E_t = energy_result['total_energy']
                    g_Q, = torch.autograd.grad(E_t, CurrQ)
                
                raw = CurrQ * torch.exp(-layer.step_size * g_Q)
                return raw / (raw.sum(dim=-1, keepdim=True) + eps)

            # Use Anderson Solver for Forward Pass
            Q_star = anderson_solver(step_fn, Q, m=5, lam=1e-4, max_iter=layer.max_steps, tol=layer.tol)
            
        ctx.save_for_backward(P, Q_star)
        ctx.layer = layer
        ctx.meta = (entity_pairs, doc_ids, e1_types, e2_types)
        ctx.doc_context = None
        if doc_context is not None:
            ctx.doc_context = doc_context
        
        if doc_context is not None:
            ctx.doc_context = doc_context
        # Compute energy outside for stats
        return Q_star

    @staticmethod
    def backward(ctx, grad_Q_star):
        P, Q_star = ctx.saved_tensors
        layer = ctx.layer
        entity_pairs, doc_ids, e1_types, e2_types = ctx.meta
        
        if getattr(ctx, 'doc_context', None) is not None:
            c = ctx.doc_context
        else:
            c = layer._build_doc_contexts(P, entity_pairs, doc_ids, e1_types, e2_types)
        
        with torch.enable_grad():
            Q_star = Q_star.detach().requires_grad_(True)
            P = P.detach().requires_grad_(True)
            
            eps = 1e-12
            def f_step(CurrQ, CurrP):
                energy_result = layer._compute_reason_energy(CurrP, CurrQ, c)
                E_t = energy_result['total_energy']
                g_Q, = torch.autograd.grad(E_t, CurrQ, create_graph=True)
                raw = CurrQ * torch.exp(-layer.step_size * g_Q)
                return raw / (raw.sum(dim=-1, keepdim=True) + eps)

            f_val = f_step(Q_star, P)
            
            # Solve v = grad_Q_star + v * J using Anderson
            # Map: v_new = grad_Q_star + v_old * J
            # This is finding fixed point of g(v) = grad_Q_star + vjp(v)
            
            def map_v(v):
                vjp, = torch.autograd.grad(f_val, Q_star, grad_outputs=v, retain_graph=True)
                return grad_Q_star + vjp
            
            # Initial guess
            v0 = grad_Q_star
            
            # Run Anderson Solver for Implicit Backward
            # Usually converges in 10-20 steps significantly better than Neumann
            v_star = anderson_solver(map_v, v0, m=5, max_iter=20, tol=1e-5)
            
            grad_P, = torch.autograd.grad(f_val, P, grad_outputs=v_star)
            
        return grad_P, None, None, None, None, None, None


class UnifiedNeuralReasoningLayer(nn.Module):
    """
    Smooth-Convex Neuro-Symbolic Reasoning Layer.

    - Rule energy: Smooth convex HL-MRF (softplus của linear distance).
    - Inference: Exponentiated-Gradient (Mirror Descent) trên simplex.
    - Q: latent logic state, xác định bởi tối ưu convex:
          Q* = argmin_Q [ E_rules(Q) + λ_ent H(Q) ]
      (KHÔNG còn term KL(Q||P); P chỉ dùng để khởi tạo Q).
    """

    def __init__(
        self,
        rule_templates: List[RuleTemplate],
        num_relations: int = 3,
        num_types: int = 11,
        step_size: float = 0.5,
        lambda_kl: float = 0.1,
        smooth_tau: float = 1.0,  # Increased from 0.1 for stronger energy signal
        max_steps: int = 50,
        tol: float = 1e-4,
        type_name_to_id: Optional[Dict[str, int]] = None,
        relation_to_index: Optional[Dict[str, int]] = None,
        predicate_registry: Optional[PredicateRegistry] = None,
        use_deq: bool = False,
        learn_rule_weights: bool = True, # [ADDED]
        initial_rule_weight: float = 0.0, # [ADDED]
        unroll_stride: int = 10, # [ADDED]
    ):
        super().__init__()
        self.use_deq = use_deq
        self.unroll_stride = unroll_stride

        self.num_relations = num_relations
        self.num_types = num_types
        self.step_size = step_size
        self.smooth_tau = smooth_tau

        self.lambda_kl = lambda_kl  # Save lambda_kl
        self.max_steps = max_steps
        self.tol = tol
        
        self.rule_templates = rule_templates
        self.num_rules = len(rule_templates)

        # Initialize rule weight logits
        # w_rules = softplus(w_rules_logits)
        self.learn_rule_weights = learn_rule_weights
        
        import math
        # softplus(0.541) ≈ 1.0
        initial_logit = math.log(math.e - 1.0)  # ≈ 0.541
        
        self.w_rules_logits = nn.Parameter(
            torch.full((len(rule_templates),), initial_logit),
            requires_grad=learn_rule_weights
        )

        # Type / relation maps
        if type_name_to_id is None:
            self.type_name_to_id = {f"TYPE{i}": i for i in range(num_types)}
        else:
            self.type_name_to_id = {k.upper(): v for k, v in type_name_to_id.items()}

        if relation_to_index is None:
            self.relation_to_index = {"BEFORE": 0, "OVERLAP": 1, "AFTER": 2, "EQUAL": 3, "VAGUE": 3}
        else:
            self.relation_to_index = {k.upper(): v for k, v in relation_to_index.items()}

        self.predicate_registry = predicate_registry or create_default_predicate_registry()

        # Compile rules
        self._compile_rules()

        # Hard rule mask
        self.hard_rule_mask = torch.zeros(self.num_rules, dtype=torch.float32)
        for i, rule in enumerate(self.rule_templates):
            if rule.is_hard:
                self.hard_rule_mask[i] = 1.0
                print(f"[UnifiedNeuralReasoningLayer] Hard Rule: {rule.name}")

        print(
            f"[UnifiedNeuralReasoningLayer] init: "
            f"rules={self.num_rules}, rel={self.num_relations}, types={self.num_types}, "
            f"smooth_tau={self.smooth_tau}, lambda_kl={self.lambda_kl}"
        )

    # --------------------------------------------------------
    # 5.1. Compile rule templates thành CompiledRule
    # --------------------------------------------------------
    def _compile_rules(self):
        self.generic_rules: List[CompiledRule] = []

        for k, rule in enumerate(self.rule_templates):
            # 1. Collect variables
            all_preds = rule.lhs + [rule.rhs]
            all_vars = []
            for p in all_preds:
                p_name = p.name.upper()
                if p_name == 'TYPE':
                    if len(p.args) > 0:
                        all_vars.append(p.args[0])
                elif p_name in ['BEFORETYPE', 'AFTERTYPE', 'OVERLAPTYPE']:
                    # Quan hệ giữa type constants, không có var
                    continue
                else:
                    all_vars.extend(p.args)

            unique_vars = sorted(list(set(all_vars)))
            var_to_idx = {v: i for i, v in enumerate(unique_vars)}
            num_vars = len(unique_vars)

            def compile_term(pred: Predicate) -> TermSpec:
                name_up = pred.name.upper()

                if name_up == "TYPE":
                    # Type(Var, TYPE_NAME)
                    if len(pred.args) < 2:
                        raise ValueError(f"Malformed TYPE predicate: {pred}")
                    var_name = pred.args[0]
                    type_name = pred.args[1].upper()
                    v_idx = var_to_idx[var_name]
                    t_idx = self.type_name_to_id.get(type_name, 0)
                    return TermSpec(source_type='T', tensor_idx=t_idx, var_indices=[v_idx], is_negated=pred.is_negated)

                # Quan hệ thời gian (Before, After, Overlap,...)
                var_indices = [var_to_idx[a] for a in pred.args]
                r_idx = self.relation_to_index.get(name_up)
                if r_idx is not None:
                    return TermSpec(source_type='M', tensor_idx=r_idx, var_indices=var_indices, is_negated=pred.is_negated)

                raise ValueError(f"Unknown predicate in rule compilation: {pred}")

            lhs_specs = [compile_term(p) for p in rule.lhs]
            rhs_spec = compile_term(rule.rhs)

            # Head vars
            head_vars = set(rhs_spec.var_indices)
            all_vars_indices = set(range(num_vars))
            mask_indices = list(all_vars_indices - head_vars)

            compiled = CompiledRule(
                rule_idx=k,
                name=rule.name, # [ADDED]
                lhs_terms=lhs_specs,
                rhs_term=rhs_spec,
                num_vars=num_vars,
                mask_indices=mask_indices,
                is_hard=rule.is_hard,
            )
            self.generic_rules.append(compiled)

    # --------------------------------------------------------
    # 5.2. Build BatchedContext
    # --------------------------------------------------------
    def build_context(self, P, entity_pairs, doc_ids, e1_types=None, e2_types=None):
        return self._build_doc_contexts(P, entity_pairs, doc_ids, e1_types, e2_types)

    def _build_doc_contexts(
        self,
        Q: torch.Tensor,
        entity_pairs: List[Tuple[int, int]],
        doc_ids: List[int],
        e1_types: Optional[torch.Tensor],
        e2_types: Optional[torch.Tensor],
    ) -> BatchedContext:
        device = Q.device

        # Group pairs by doc
        doc_to_indices: Dict[Any, List[int]] = {}
        for idx, d in enumerate(doc_ids):
            d_key = d
            if d_key not in doc_to_indices:
                doc_to_indices[d_key] = []
            doc_to_indices[d_key].append(idx)

        unique_docs = list(doc_to_indices.keys())
        batch_size = len(unique_docs)
        doc_to_batch_idx = {d: i for i, d in enumerate(unique_docs)}

        # Entities per doc
        doc_entities = {}
        doc_ent_to_idx = {}
        max_ent = 0
        doc_ent_types = {}

        for d_key in unique_docs:
            idxs = doc_to_indices[d_key]
            ent_type_map = {}
            if e1_types is not None:
                for g_idx in idxs:
                    e1, e2 = entity_pairs[g_idx]
                    t1 = int(e1_types[g_idx].item())
                    t2 = int(e2_types[g_idx].item())
                    ent_type_map[e1] = t1
                    ent_type_map[e2] = t2
            doc_ent_types[d_key] = ent_type_map

            entities = set(ent_type_map.keys())
            if not entities:
                for g_idx in idxs:
                    e1, e2 = entity_pairs[g_idx]
                    entities.add(e1)
                    entities.add(e2)

            sorted_ents = sorted(list(entities))
            doc_entities[d_key] = sorted_ents
            doc_ent_to_idx[d_key] = {e: i for i, e in enumerate(sorted_ents)}
            max_ent = max(max_ent, len(sorted_ents))

        if max_ent == 0:
            return BatchedContext(
                scatter_indices=(
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                ),
                global_indices=torch.empty(0, dtype=torch.long, device=device),
                T_mask=None,
                entity_mask=torch.zeros(batch_size, 0, device=device),
                batch_size=batch_size,
                max_ent=0,
            )

        # Scatter indices + masks
        batch_indices_list = []
        row_indices_list = []
        col_indices_list = []
        global_indices_list = []

        entity_mask = torch.zeros(batch_size, max_ent, device=device)
        pair_mask = torch.zeros(batch_size, max_ent, max_ent, device=device) # [ADDED]

        if e1_types is not None:
            T_mask = torch.zeros(batch_size, max_ent, self.num_types, device=device)
        else:
            T_mask = None

        for d_key in unique_docs:
            b_idx = doc_to_batch_idx[d_key]
            local_ent_map = doc_ent_to_idx[d_key]
            num_local = len(local_ent_map)

            entity_mask[b_idx, :num_local] = 1.0

            if T_mask is not None:
                type_map = doc_ent_types[d_key]
                for e, t_id in type_map.items():
                    if e in local_ent_map and 0 <= t_id < self.num_types:
                        idx = local_ent_map[e]
                        T_mask[b_idx, idx, t_id] = 1.0

            idxs = doc_to_indices[d_key]
            for g_idx in idxs:
                e1, e2 = entity_pairs[g_idx]
                if e1 in local_ent_map and e2 in local_ent_map:
                    r = local_ent_map[e1]
                    c = local_ent_map[e2]
                    batch_indices_list.append(b_idx)
                    row_indices_list.append(r)
                    col_indices_list.append(c)
                    global_indices_list.append(g_idx)
                    pair_mask[b_idx, r, c] = 1.0 # [ADDED]

        scatter_indices = (
            torch.tensor(batch_indices_list, device=device, dtype=torch.long),
            torch.tensor(row_indices_list, device=device, dtype=torch.long),
            torch.tensor(col_indices_list, device=device, dtype=torch.long),
        )
        global_indices = torch.tensor(global_indices_list, device=device, dtype=torch.long)

        return BatchedContext(
            scatter_indices=scatter_indices,
            global_indices=global_indices,
            T_mask=T_mask,
            entity_mask=entity_mask,
            pair_mask=pair_mask, # [ADDED]
            batch_size=batch_size,
            max_ent=max_ent,
        )


    # --------------------------------------------------------
    # 5.3. Smooth HL-MRF Rule Energy: E_rules(Q)
    # --------------------------------------------------------
    def _evaluate_single_rule(
        self,
        rule,
        M: torch.Tensor,
        T_mask: torch.Tensor,
        broadcast_term,
        tau: float,
        ctx: BatchedContext,
        device,
        rule_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate a single rule and return its energy contribution and statistics.
        
        Args:
            rule: The rule to evaluate
            M: Relation tensor [B, N, N, R]
            T_mask: Type mask tensor [B, N, num_types]
            broadcast_term: Function to broadcast terms to correct shape
            tau: Smoothing temperature
            ctx: Batched context
            device: Torch device
            rule_weights: Tensor of rule weights
            
        Returns:
            Dictionary with: energy, grounding_count, violation_count, weight
        """
        # Compute LHS sum
        lhs_sum = torch.tensor(0.0, device=device)
        for term in rule.lhs_terms:
            if term.source_type == 'M': # Relation predicate
                T_rel = M[..., term.tensor_idx]
            else: # Type predicate
                T_rel = T_mask[..., term.tensor_idx]
            
            T_broad = broadcast_term(T_rel, term.var_indices, rule.num_vars)
            if term.is_negated:
                T_broad = 1.0 - T_broad
            lhs_sum = lhs_sum + T_broad
        
        # Compute RHS term
        rhs_term = rule.rhs_term
        if rhs_term.source_type == 'M':
            R_tensor = M[..., rhs_term.tensor_idx]
        else:
            R_tensor = T_mask[..., rhs_term.tensor_idx]
        rhs_broad = broadcast_term(R_tensor, rhs_term.var_indices, rule.num_vars)
        if rhs_term.is_negated:
            rhs_broad = 1.0 - rhs_broad
        
        # Linear combination for the rule
        offset = len(rule.lhs_terms) - 1.0
        lin = lhs_sum - rhs_broad - offset
        
        # Smooth hinge loss
        hinge_smooth = F.softplus(lin / tau) * tau
        
        # Check absolute violations (threshold > 1e-6 to capture micro-violations)
        is_violated = (hinge_smooth > 1e-6).float()
        
        # Apply grounding mask based on entity and pair existence
        accum_mask = None
        # 1. Ensure all variables are real entities
        for v_idx in range(rule.num_vars):
            m = ctx.entity_mask  # [B, N]
            m_broad = broadcast_term(m, [v_idx], rule.num_vars)
            if accum_mask is None:
                accum_mask = m_broad
            else:
                accum_mask = accum_mask * m_broad

        # 2. Ensure all relation atoms in the rule exist as candidate pairs
        for term in rule.lhs_terms + [rule.rhs_term]:
            if term.source_type == 'M':
                p_mask = broadcast_term(ctx.pair_mask, term.var_indices, rule.num_vars)
                if accum_mask is None: # Should not happen if rule has vars
                    accum_mask = p_mask
                else:
                    accum_mask = accum_mask * p_mask
            
        grounding_count = 0.0
        violation_count = 0.0
        
        if accum_mask is not None:
            # Apply mask to violations and energy
            hinge_smooth = hinge_smooth * accum_mask
            is_violated = is_violated * accum_mask
            
            grounding_count = accum_mask.sum()
            violation_count = is_violated.sum()
        else:
            # No mask, all groundings are valid
            grounding_count = torch.tensor(hinge_smooth.numel(), device=device, dtype=hinge_smooth.dtype)
            violation_count = is_violated.sum()
        
        # Compute weighted energy contribution
        # IMPORTANT: Keep as Tensor for gradient flow
        energy = rule_weights[rule.rule_idx] * hinge_smooth.sum()
        
        return {
            'energy': energy,  # Tensor (for gradients)
            'grounding_count': float(grounding_count.item()),
            'violation_count': float(violation_count.item()),
            'weight': rule_weights[rule.rule_idx].item()
        }

    def _compute_rule_energy(self, Q: torch.Tensor, ctx: BatchedContext) -> Dict[str, Any]:
        """
        Smooth convex HL-MRF:
            l_r(Q) = sum(A_i) - C - (k-1)
            phi_r(Q) = softplus(l_r / tau) * tau
            E_rules = sum_r w_r * E[phi_r(Q)^2]

        Returns:
            A dictionary containing:
            - 'energy': The total rule energy.
            - 'total_groundings': Total number of groundings across all rules.
            - 'violation_count': Total number of violations across all rules.
            - 'rule_details': A dictionary with per-rule statistics.
        """
        device = Q.device
        
        total_energy = torch.tensor(0.0, device=device)
        total_groundings = 0.0
        total_violations = 0.0
        rule_details = {}

        if ctx.max_ent == 0 or len(self.generic_rules) == 0:
            return {
                "energy": total_energy,
                "total_groundings": total_groundings,
                "violation_count": total_violations,
                "rule_details": rule_details
            }

        # Compute w_rules using softplus
        w_rules = F.softplus(self.w_rules_logits)
        hard_mask = self.hard_rule_mask.to(device)
        rule_weights = w_rules * (1.0 - hard_mask) + 1.0 * hard_mask # Hard rules have weight 1.0

        # Build M Matrix (Relation Tensor)
        B = ctx.batch_size
        N = ctx.max_ent
        R = self.num_relations
        M = torch.zeros(B, N, N, R, device=device)
        b_idx, r_idx, c_idx = ctx.scatter_indices
        
        # Fill Direct Pairs from Q
        if len(b_idx) > 0:
            M[b_idx, r_idx, c_idx, :] = Q[ctx.global_indices, :]
        
        # Type Mask
        if ctx.T_mask is not None:
            T_mask = ctx.T_mask
        else:
            T_mask = torch.zeros(B, N, self.num_types, device=device)
            
        def broadcast_term(tensor: torch.Tensor, term_vars: List[int], num_vars: int) -> torch.Tensor:
            B_local = tensor.shape[0]
            if len(term_vars) == 0: # Constant term, broadcast to all dimensions
                 shape = [B_local] + [1] * num_vars
                 return tensor.view(*shape).expand(B_local, *[N]*num_vars)
            
            # Reorder dimensions if necessary to match var_indices order
            if len(term_vars) > 1:
                p = sorted(range(len(term_vars)), key=lambda k: term_vars[k])
                if p != list(range(len(term_vars))):
                    full_perm = [0] + [x + 1 for x in p] # Batch dim + var dims
                    tensor = tensor.permute(full_perm)
            
            current_vars = sorted(term_vars)
            slices = [slice(None)] # For batch dimension
            var_ptr = 0
            for k in range(num_vars):
                if var_ptr < len(current_vars) and current_vars[var_ptr] == k:
                    slices.append(slice(None)) # This variable is present
                    var_ptr += 1
                else:
                    slices.append(None) # This variable is not present, add a new dimension
            return tensor[tuple(slices)]
            
        tau = self.smooth_tau
        
        for rule in self.generic_rules:
            rule_result = self._evaluate_single_rule(
                rule, M, T_mask, broadcast_term, tau, ctx, device, rule_weights
            )
            
            # Accumulate totals
            if rule_result['grounding_count'] > 0:
                total_groundings += rule_result['grounding_count']
                total_violations += rule_result['violation_count']
                total_energy = total_energy + rule_result['energy']
                
            # Store per-rule stats
            rule_details[rule.name] = {
                'weight': rule_result['weight'],
                'grounding_count': rule_result['grounding_count'],
                'violation_count': rule_result['violation_count']
            }
            
        return {
            "energy": total_energy,
            "total_groundings": total_groundings,
            "violation_count": total_violations,
            "rule_details": rule_details
        }

    def compute_stats(self, Q: torch.Tensor, ctx: BatchedContext, step: int = -1) -> Dict[str, float]:
        """
        Wrapper function for backward compatibility.
        Calls _compute_rule_energy and returns the same dictionary.
        """
        return self._compute_rule_energy(Q, ctx)

    def _compute_reason_energy(self, P: torch.Tensor, Q: torch.Tensor, ctx: BatchedContext) -> Dict[str, Any]:
        """
        P, Q: [Total_Pairs, num_relations]
        P acts as the 'Unary Potential' or Prior.
        
        Returns:
            Dictionary with:
            - 'total_energy': Scalar tensor (for gradients)
            - 'energy_rules': Rule energy component
            - 'energy_entropy': Entropy component  
            - 'energy_kl': KL component
            - 'total_groundings': Total groundings count
            - 'violation_count': Total violations
            - 'rule_details': Per-rule statistics
        """
        device = Q.device
        eps = 1e-12

        # 1. E_rules(Q) - returns a dictionary
        rule_result = self._compute_rule_energy(Q, ctx)
        E_rules = rule_result['energy']

        # 2. Entropy term (Minimize Entropy => Encourage Certainty)
        E_ent = - (Q * torch.log(Q + eps)).sum(dim=-1).mean()

        # 3. KL(Q || P) - Standard anchor term to prevent Q from deviating too far from P
        # Formula: KL(Q || P) = sum_k Q_k * log(Q_k / P_k)
        # This penalizes Q for assigning high probability where P assigns low probability
        E_kl = (Q * (torch.log(Q + eps) - torch.log(P + eps))).sum(dim=-1).mean()

        total_energy = E_rules + self.lambda_kl * E_kl
        
        return {
            'total_energy': total_energy,
            'energy_rules': E_rules,
            'energy_entropy': E_ent,
            'energy_kl': E_kl,
            'total_groundings': rule_result['total_groundings'],
            'violation_count': rule_result['violation_count'],
            'rule_details': rule_result['rule_details']
        }

    # --------------------------------------------------------
    # 5.5. Inference: Mirror Descent (Exponentiated Gradient)
    # --------------------------------------------------------
    def infer_Q(
        self,
        logits: torch.Tensor,
        entity_pairs: List[Tuple[int, int]],
        doc_ids,
        e1_types: Optional[torch.Tensor] = None,
        e2_types: Optional[torch.Tensor] = None,
        ctx: Optional[BatchedContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Original Version (restored): Unrolled optimization loop.
        Simple, debuggable, but O(N) memory.
        """
        if isinstance(doc_ids, torch.Tensor):
            doc_ids_list = doc_ids.detach().cpu().tolist()
        else:
            doc_ids_list = list(doc_ids)

        device = logits.device
        eps = 1e-12
        P = F.softmax(logits, dim=-1)

        if ctx is None:
            ctx = self._build_doc_contexts(P, entity_pairs, doc_ids_list, e1_types, e2_types)
        
        Q = P.clone()
        step = 0
        prev_Q = Q
        energy_history = []  # Store energy trajectory for analysis
    
        with torch.enable_grad():
            if not Q.requires_grad:
                Q.requires_grad_(True)

            while step < self.max_steps:
                is_grad_step = ((step + 1) % self.unroll_stride == 0) or (step == self.max_steps - 1)
                
                energy_result = self._compute_reason_energy(P, Q, ctx)
                E_total = energy_result['total_energy']
                
                # Store complete energy snapshot (detach tensors to save memory)
                energy_history.append({
                    'step': step,
                    'total_energy': E_total.item(),
                    'energy_rules': energy_result['energy_rules'].item(),
                    'energy_entropy': energy_result['energy_entropy'].item(),
                    'energy_kl': energy_result['energy_kl'].item(),
                    'violation_count': energy_result['violation_count'],
                    'total_groundings': energy_result['total_groundings'],
                    'rule_details': energy_result['rule_details']  # Store rule details too!
                })
                
                grad_Q, = torch.autograd.grad(E_total, Q, create_graph=is_grad_step, retain_graph=True)
                grad_Q = grad_Q.clamp(min=-10.0, max=10.0)
                
                Q = Q.clamp(min=1e-7)
                
                Q_new = Q * torch.exp(-self.step_size * grad_Q)
                Q_new = Q_new / (Q_new.sum(dim=-1, keepdim=True) + eps)
                Q_new = Q_new.clamp(min=1e-7)
        

                diff = torch.norm(Q_new - prev_Q).item()
                prev_Q = Q_new
                Q = Q_new

                step += 1
                if diff < self.tol:
                    break
    
        stats = {
            'energy_history': energy_history
        }
        return Q, stats

    def infer_Q_deq(
        self,
        logits: torch.Tensor,
        entity_pairs: List[Tuple[int, int]],
        doc_ids,
        e1_types: Optional[torch.Tensor] = None,
        e2_types: Optional[torch.Tensor] = None,
        ctx: Optional[BatchedContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implicit Differentiation Version.
        O(1) Memory, O(1) Gradient Path.
        """
        if isinstance(doc_ids, torch.Tensor):
            doc_ids_list = doc_ids.detach().cpu().tolist()
        else:
            doc_ids_list = list(doc_ids)
            
        P = F.softmax(logits, dim=-1)
        
        # Ensure Context exists beforehand
        if ctx is None:
             ctx = self._build_doc_contexts(
                 P, entity_pairs, doc_ids_list, e1_types, e2_types
             )

        # Call the Autograd Function
        # Now returns only Q_star
        Q_star = ImplicitReasoningFunction.apply(
            P, 
            self, 
            entity_pairs, 
            doc_ids_list, 
            e1_types, 
            e2_types,
            ctx
        )
        
        # Compute stats for BOTH P and Q
        stats_P = self.compute_stats(P, ctx)
        stats_Q = self.compute_stats(Q_star, ctx)
        
        # Combine stats
        stats = {
            'energy_P': stats_P['energy'],
            'energy_Q': stats_Q['energy'],
            'rule_details_P': stats_P['rule_details'],
            'rule_details_Q': stats_Q['rule_details'],
            'violation_count_P': stats_P['violation_count'],
            'violation_count_Q': stats_Q['violation_count'],
            'total_groundings': stats_Q['total_groundings'],
        }
        
        return Q_star, stats


    
    # --------------------------------------------------------
    # 5.6. PSL-style rule loss trên P (cho Baseline+PSL)
    # --------------------------------------------------------
    def compute_rule_loss_with_probs(
        self,
        probs: torch.Tensor,
        entity_pairs: List[Tuple[int, int]],
        doc_ids,
        e1_types: Optional[torch.Tensor] = None,
        e2_types: Optional[torch.Tensor] = None,
        ctx: Optional[BatchedContext] = None,
    ) -> torch.Tensor:
        """
        Dùng cho kịch bản Baseline+PSL:
        - P = probs
        - Rule loss = E_rules(P) (KHÔNG KL, KHÔNG entropy)
        """
        if isinstance(doc_ids, torch.Tensor):
            doc_ids_list = doc_ids.detach().cpu().tolist()
        else:
            doc_ids_list = list(doc_ids)


        if ctx is None:
            ctx = self._build_doc_contexts(
                Q=probs,
                entity_pairs=entity_pairs,
                doc_ids=doc_ids_list,
                e1_types=e1_types,
                e2_types=e2_types,
            )
        return self._compute_rule_energy(probs, ctx)

    # --------------------------------------------------------
    # 5.7. forward = alias cho infer_Q (tương thích với code cũ)
    # --------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        entity_pairs: List[Tuple[int, int]],
        doc_ids,
        e1_types: Optional[torch.Tensor] = None,
        e2_types: Optional[torch.Tensor] = None,
        ctx: Optional[BatchedContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified Forward: Dispatch based on configuration.
        """
        if self.use_deq:
            return self.infer_Q_deq(
                logits=logits,
                entity_pairs=entity_pairs,
                doc_ids=doc_ids,
                e1_types=e1_types,
                e2_types=e2_types,
                ctx=ctx,
            )
        else:
            return self.infer_Q(
                logits=logits,
                entity_pairs=entity_pairs,
                doc_ids=doc_ids,
                e1_types=e1_types,
                e2_types=e2_types,
                ctx=ctx,
            )
