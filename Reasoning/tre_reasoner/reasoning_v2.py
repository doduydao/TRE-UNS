import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
import re

# Reuse GradientReversal from v1
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ============================================================
# [PORTED] Anderson Solver & Implicit Function
# ============================================================

def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0):
    """
    Anderson acceleration for fixed point iteration: x = f(x).
    """
    bsz = x0.shape[0]
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
            G = torch.stack(G, dim=1) 
            
            GtG = torch.matmul(G.t(), G) # [n, n]
            GtG = GtG + lam * torch.eye(n, device=GtG.device)
            
            try:
                # Solve H * alpha = 1
                ones = torch.ones(n, 1, device=GtG.device)
                alpha_unc = torch.linalg.solve(GtG, ones)
                alpha = alpha_unc / alpha_unc.sum()
                alpha = alpha.squeeze() # [n]
                
                # x_{k+1} = \sum alpha_i F_i
                x_new = torch.zeros_like(x)
                if n > 1 and alpha.ndim > 0:
                   for i in range(n):
                       x_new += alpha[i] * F[i]
                else:
                   x_new = F[-1] # Fallback
                x = x_new
            except:
                x = fx * beta + x * (1 - beta)
        
    return x.view_as(x0)


class ImplicitReasoningFunctionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, weights, layer, entity_pairs, doc_ids, e1_types, e2_types, doc_context=None):
        # 1. Forward: Find Fixed Point Q*
        with torch.no_grad():
            if doc_context is not None:
                c = doc_context
            else:
                c = layer.build_context(P, entity_pairs, doc_ids, e1_types, e2_types)
            
            Q = P.clone()
            
            # Define step function for solver
            eps = 1e-12
            def step_fn(CurrQ):
                # Gradient descent step
                # IMPORTANT: enable_grad needed for dE/dQ even inside no_grad block
                with torch.enable_grad():
                    CurrQ = CurrQ.detach().requires_grad_(True)
                    energy_result = layer._compute_reason_energy(P, CurrQ, c)
                    E_re = energy_result['E_re']
                    # Use create_graph=False here, we just want gradient value for update
                    g_Q, = torch.autograd.grad(E_re, CurrQ)
                    clip_val = getattr(layer, "grad_clip_value", 10.0)
                    g_Q = torch.clamp(g_Q, min=-clip_val, max=clip_val) # [STABILITY] Clamp gradients
                
                # Log-Space Mirror Descent Step (Stable)
                log_q = torch.log(CurrQ + eps)
                log_q_new = log_q - layer.step_size * g_Q
                return F.softmax(log_q_new, dim=-1)

            # Use Anderson Solver for Forward Pass
            Q_star = anderson_solver(step_fn, Q, m=5, lam=1e-4, max_iter=layer.max_steps, tol=layer.tol)
            
        ctx.save_for_backward(P, Q_star, weights)
        ctx.layer = layer
        ctx.meta = (entity_pairs, doc_ids, e1_types, e2_types)
        if doc_context is not None:
            ctx.doc_context = doc_context
        else:
            ctx.doc_context = None

        return Q_star

    @staticmethod
    def backward(ctx, grad_Q_star):
        P, Q_star, weights = ctx.saved_tensors
        layer = ctx.layer
        entity_pairs, doc_ids, e1_types, e2_types = ctx.meta
        
        if getattr(ctx, 'doc_context', None) is not None:
            c = ctx.doc_context
        else:
            c = layer.build_context(P, entity_pairs, doc_ids, e1_types, e2_types)
        
        with torch.enable_grad():
            Q_star = Q_star.detach().requires_grad_(True)
            P = P.detach().requires_grad_(True)
            w = weights.detach().requires_grad_(True)
            
            eps = 1e-12
            def f_step(CurrQ, CurrP, CurrW):
                energy_result = layer._compute_reason_energy(CurrP, CurrQ, c, w_logits=CurrW)
                E_re = energy_result['E_re']
                g_Q, = torch.autograd.grad(E_re, CurrQ, create_graph=True)
                clip_val = getattr(layer, "grad_clip_value", 10.0)
                g_Q = torch.clamp(g_Q, min=-clip_val, max=clip_val)
                log_Q = torch.log(CurrQ + eps) - layer.step_size * g_Q
                return F.softmax(log_Q, dim=-1)

            # Recompute f(Q*, P, w) inside map_v to avoid retaining large graphs
            def map_v(v):
                f_val = f_step(Q_star, P, w)
                vjp, = torch.autograd.grad(f_val, Q_star, grad_outputs=v, retain_graph=False)
                return grad_Q_star + vjp
            
            v0 = grad_Q_star.clone() 
            v_star = anderson_solver(map_v, v0, m=5, max_iter=layer.max_steps, tol=1e-5)
            
            # Compute gradients w.r.t P and W (recompute to avoid retaining graph)
            f_val = f_step(Q_star, P, w)
            grad_P, grad_W = torch.autograd.grad(f_val, (P, w), grad_outputs=v_star, retain_graph=False, allow_unused=True)
                     
        return grad_P, grad_W, None, None, None, None, None, None, None


# ============================================================
# 1. Generalized FOL AST
# ============================================================

class LogicNode:
    """Base class for all logical expression nodes."""
    def vars(self) -> List[str]:
        raise NotImplementedError()

@dataclass
class AtomNode(LogicNode):
    """
    Leaf node: Predicate(V1, V2, ...)
    """
    name: str
    args: List[str]
    is_negated: bool = False

    def vars(self) -> List[str]:
        # For Type(E1, PROBLEM), only E1 is a variable
        if self.name.upper() == "TYPE":
            return [self.args[0]]
        # For TimeRel(A, B, R) and PredTimeRel(A, B, R), only A and B are variables
        if self.name.upper() in ["TIMEREL", "PREDTIMEREL"]:
            return self.args[:2]
        return self.args

@dataclass
class UnaryOpNode(LogicNode):
    """
    Unary operator: !Node
    """
    op: str # '!'
    child: LogicNode

    def vars(self) -> List[str]:
        return self.child.vars()

@dataclass
class BinaryOpNode(LogicNode):
    """
    Binary operator: Node1 OP Node2
    """
    op: str # '&', '|', '=>', '<=>'
    left: LogicNode
    right: LogicNode

    def vars(self) -> List[str]:
        return sorted(list(set(self.left.vars() + self.right.vars())))

@dataclass
class RuleTemplateV2:
    """
    General FOL Rule.
    """
    name: str
    formula: LogicNode
    is_hard: bool = False

# ============================================================
# 2. Logic Parser
# ============================================================

PRED_PATTERN = re.compile(r'\s*([!~]?)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*')

def parse_predicate_to_node(text: str) -> AtomNode:
    m = PRED_PATTERN.match(text)
    if not m:
        raise ValueError(f"Cannot parse atom: {text}")
    neg_char = m.group(1)
    name = m.group(2)
    args_str = m.group(3)
    args = [a.strip() for a in args_str.split(",") if a.strip()]
    is_negated = (neg_char in ['!', '~'])
    return AtomNode(name=name, args=args, is_negated=is_negated)

class LogicParser:
    """
    Simplistic recursive descent parser for logic formulas.
    Precedence: ! > & > | > => > <=>
    """
    def __init__(self, text: str):
        # Normalize: replace symbols for easier parsing
        text = text.replace('&&', '&').replace('||', '|')
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize including parenthesies and operators
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c in '()!&|':
                tokens.append(c)
                i += 1
            elif text[i:i+2] == '=>':
                tokens.append('=>')
                i += 2
            elif text[i:i+3] == '<=>':
                tokens.append('<=>')
                i += 3
            else:
                # Catch predicate or word
                start = i
                # If it's a predicate, it has '(' soon
                paren_depth = 0
                found_at_least_one_paren = False
                while i < len(text):
                    if text[i] == '(':
                        paren_depth += 1
                        found_at_least_one_paren = True
                    elif text[i] == ')':
                        paren_depth -= 1
                    
                    # Stop if we hit an operator outside parens
                    if paren_depth == 0:
                        if found_at_least_one_paren:
                            i += 1 # Include the closing paren
                            break
                        if text[i] in ' !&|=': # Potential operator start
                             break
                    i += 1
                tokens.append(text[start:i].strip())
        return tokens

    def parse(self) -> LogicNode:
        return self._parse_equiv()

    def _parse_equiv(self) -> LogicNode:
        node = self._parse_implies()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '<=>':
            self.pos += 1
            right = self._parse_implies()
            node = BinaryOpNode('<=>', node, right)
        return node

    def _parse_implies(self) -> LogicNode:
        node = self._parse_or()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '=>':
            self.pos += 1
            right = self._parse_or()
            node = BinaryOpNode('=>', node, right)
        return node

    def _parse_or(self) -> LogicNode:
        node = self._parse_and()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '|':
            self.pos += 1
            right = self._parse_and()
            node = BinaryOpNode('|', node, right)
        return node

    def _parse_and(self) -> LogicNode:
        node = self._parse_unary()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '&':
            self.pos += 1
            right = self._parse_unary()
            node = BinaryOpNode('&', node, right)
        return node

    def _parse_unary(self) -> LogicNode:
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '!':
            self.pos += 1
            return UnaryOpNode('!', self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> LogicNode:
        token = self.tokens[self.pos]
        if token == '(':
            self.pos += 1
            node = self.parse()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return node
        else:
            self.pos += 1
            return parse_predicate_to_node(token)

def parse_rule_line_v2(line: str) -> Optional[RuleTemplateV2]:
    if "#" in line:
        line = line.split("#", 1)[0]
    line = line.strip()
    if not line:
        return None

    if ":" not in line:
        raise ValueError(f"Missing rule name: {line}")
    name_part, rest = line.split(":", 1)
    rule_name = name_part.strip()

    is_hard = False
    if "[HARD]" in rest:
        is_hard = True
        rest = rest.replace("[HARD]", "").strip()

    parser = LogicParser(rest)
    formula = parser.parse()
    return RuleTemplateV2(name=rule_name, formula=formula, is_hard=is_hard)

# ============================================================
# 3. Compilation & Evaluation structures (Vectorized)
# ============================================================

@dataclass
class TermSpec:
    source_tensor: str  # 'T', 'M_PRED', 'M_GT'
    tensor_idx: int
    var_indices: List[int]
    is_negated: bool = False

class CompiledLogicNode:
    pass

@dataclass
class CompiledAtom(CompiledLogicNode):
    spec: TermSpec

@dataclass
class CompiledUnary(CompiledLogicNode):
    op: str
    child: CompiledLogicNode

@dataclass
class CompiledBinary(CompiledLogicNode):
    op: str
    left: CompiledLogicNode
    right: CompiledLogicNode

@dataclass
class CompiledRuleV2:
    rule_idx: int
    name: str
    root: CompiledLogicNode
    num_vars: int
    is_hard: bool = False
    rel_atoms: List[CompiledAtom] = field(default_factory=list)

@dataclass
class BatchedContext:
    """
    Context for the batch.
    """
    scatter_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    global_indices: torch.Tensor
    T_mask: Optional[torch.Tensor]
    entity_mask: torch.Tensor
    pair_mask: torch.Tensor
    batch_size: int
    max_ent: int

@dataclass
class PrecomputedRuleData:
    accum_mask: torch.Tensor
    constant_atoms: Dict[int, torch.Tensor]

class UnifiedNeuralReasoningLayerV2(nn.Module):
    """
    Generalized Smooth-Convex Neuro-Symbolic Reasoning Layer (V2).
    Supports arbitrary FOL expressions via recursive evaluation.
    """

    def __init__(
        self,
        rule_templates: List[RuleTemplateV2],
        num_relations: int = 3,
        num_types: int = 11,
        step_size: float = 0.5,
        lambda_kl: float = 0.1,
        lambda_entropy: float = 0.0,
        smooth_tau: float = 1.0,
        max_steps: int = 50,
        tol: float = 1e-4,
        type_name_to_id: Optional[Dict[str, int]] = None,
        relation_to_index: Optional[Dict[str, int]] = None,
        use_deq: bool = False,
        learn_rule_weights: bool = True,
        rule_chunk_size: Optional[int] = 1,
    ):
        super().__init__()
        self.use_deq = use_deq
        self.num_relations = num_relations
        self.num_types = num_types
        self.step_size = step_size
        self.smooth_tau = smooth_tau
        self.lambda_kl = lambda_kl
        self.lambda_entropy = lambda_entropy
        self.max_steps = max_steps
        self.tol = tol
        self.grad_clip_value = 10.0
        self.rule_chunk_size = rule_chunk_size
        self.rule_templates = rule_templates
        self.num_rules = len(rule_templates)

        import math
        initial_logit = math.log(math.e - 1.0)
        self.w_rules_logits = nn.Parameter(
            torch.full((len(rule_templates),), initial_logit),
            requires_grad=learn_rule_weights
        )

        if type_name_to_id is None:
            self.type_name_to_id = {f"TYPE{i}": i for i in range(num_types)}
        else:
            self.type_name_to_id = {k.upper(): v for k, v in type_name_to_id.items()}

        if relation_to_index is None:
            # Default to MATRES standard (0:BEFORE, 1:AFTER, 2:EQUAL, 3:VAGUE)
            self.relation_to_index = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
        else:
            self.relation_to_index = {k.upper(): v for k, v in relation_to_index.items()}

        self.inverse_rel_map = self._infer_inverse_map()
        self._compile_rules()
        print(f"DEBUG REASONING: Initialized with {self.num_rules} rules. First rule: {self.rule_templates[0].name if self.num_rules > 0 else 'NONE'}")

        self.hard_rule_mask = torch.zeros(self.num_rules, dtype=torch.float32)
        for i, rule in enumerate(self.rule_templates):
            if rule.is_hard:
                self.hard_rule_mask[i] = 1.0
        
        self.cnt_debug = 0

    def _infer_inverse_map(self) -> List[int]:
        inv_map = list(range(self.num_relations))
        pairs = [("BEFORE", "AFTER"), ("AFTER", "BEFORE")]
        for a, b in pairs:
            if a in self.relation_to_index and b in self.relation_to_index:
                inv_map[self.relation_to_index[a]] = self.relation_to_index[b]
        return inv_map

    @staticmethod
    def _to_cpu_scalar(x, default=0.0):
        if torch.is_tensor(x):
            return x.detach().item()
        if x is None:
            return default
        return x

    @staticmethod
    def _collect_rel_atoms(node: CompiledLogicNode) -> List[CompiledAtom]:
        atoms = []
        if isinstance(node, CompiledAtom):
            if node.spec.source_tensor in ['M_PRED', 'M_GT'] and len(node.spec.var_indices) == 2:
                atoms.append(node)
        elif isinstance(node, CompiledUnary):
            atoms.extend(UnifiedNeuralReasoningLayerV2._collect_rel_atoms(node.child))
        elif isinstance(node, CompiledBinary):
            atoms.extend(UnifiedNeuralReasoningLayerV2._collect_rel_atoms(node.left))
            atoms.extend(UnifiedNeuralReasoningLayerV2._collect_rel_atoms(node.right))
        return atoms

    def _compile_rules(self):
        self.compiled_rules: List[CompiledRuleV2] = []
        for k, rule in enumerate(self.rule_templates):
            unique_vars = rule.formula.vars()
            var_to_idx = {v: i for i, v in enumerate(unique_vars)}
            num_vars = len(unique_vars)

            def compile_node(node: LogicNode) -> CompiledLogicNode:
                if isinstance(node, AtomNode):
                    name_up = node.name.upper()
                    if name_up == "TYPE":
                        var_name = node.args[0]
                        type_name = node.args[1].upper()
                        v_idx = var_to_idx[var_name]
                        t_idx = self.type_name_to_id.get(type_name, 0)
                        spec = TermSpec(source_tensor='T', tensor_idx=t_idx, var_indices=[v_idx], is_negated=node.is_negated)
                    elif name_up == "TIMEREL":
                        v1, v2, rel_name = node.args
                        v_indices = [var_to_idx[v1], var_to_idx[v2]]
                        r_idx = self.relation_to_index.get(rel_name.upper(), 0)
                        spec = TermSpec(source_tensor='M_GT', tensor_idx=r_idx, var_indices=v_indices, is_negated=node.is_negated)
                    elif name_up == "PREDTIMEREL":
                        v1, v2, rel_name = node.args
                        v_indices = [var_to_idx[v1], var_to_idx[v2]]
                        r_idx = self.relation_to_index.get(rel_name.upper(), 0)
                        spec = TermSpec(source_tensor='M_PRED', tensor_idx=r_idx, var_indices=v_indices, is_negated=node.is_negated)
                    else:
                        var_indices = [var_to_idx[a] for a in node.args]
                        r_idx = self.relation_to_index.get(name_up, 0)
                        spec = TermSpec(source_tensor='M_PRED', tensor_idx=r_idx, var_indices=var_indices, is_negated=node.is_negated)
                    return CompiledAtom(spec=spec)
                elif isinstance(node, UnaryOpNode):
                    return CompiledUnary(node.op, compile_node(node.child))
                elif isinstance(node, BinaryOpNode):
                    return CompiledBinary(node.op, compile_node(node.left), compile_node(node.right))
                raise ValueError(f"Unknown node type: {type(node)}")

            root = compile_node(rule.formula)
            rel_atoms = self._collect_rel_atoms(root)
            self.compiled_rules.append(CompiledRuleV2(
                rule_idx=k, name=rule.name, root=root, num_vars=num_vars, is_hard=rule.is_hard, rel_atoms=rel_atoms
            ))

    def _broadcast_term(self, tensor: torch.Tensor, term_vars: List[int], num_vars: int, N: int) -> torch.Tensor:
        B_local = tensor.shape[0]
        if len(term_vars) == 0:
             return tensor.view(B_local, *[1]*num_vars).expand(B_local, *[N]*num_vars)
        if len(term_vars) > 1:
            p = sorted(range(len(term_vars)), key=lambda k: term_vars[k])
            if p != list(range(len(term_vars))):
                full_perm = [0] + [x + 1 for x in p]
                tensor = tensor.permute(full_perm)
        current_vars = sorted(term_vars)
        slices = [slice(None)]
        var_ptr = 0
        for k in range(num_vars):
            if var_ptr < len(current_vars) and current_vars[var_ptr] == k:
                slices.append(slice(None))
                var_ptr += 1
            else:
                slices.append(None)
        return tensor[tuple(slices)]

    def _evaluate_recursive(self, node: CompiledLogicNode, M_pred, M_gt, T_mask, num_vars, N, tau, pre_data: Optional[PrecomputedRuleData] = None) -> torch.Tensor:
        if isinstance(node, CompiledAtom):
            if pre_data is not None and id(node) in pre_data.constant_atoms:
                return pre_data.constant_atoms[id(node)]
            spec = node.spec
            if spec.source_tensor == 'M_PRED':
                val = M_pred[..., spec.tensor_idx]
            elif spec.source_tensor == 'M_GT':
                val = M_gt[..., spec.tensor_idx] if M_gt is not None else torch.zeros_like(M_pred[..., 0])
            else: # 'T'
                val = T_mask[..., spec.tensor_idx]
            return self._broadcast_term(val, spec.var_indices, num_vars, N)
        elif isinstance(node, CompiledUnary):
            child_val = self._evaluate_recursive(node.child, M_pred, M_gt, T_mask, num_vars, N, tau, pre_data)
            if node.op == '!': return 1.0 - child_val
        elif isinstance(node, CompiledBinary):
            left = self._evaluate_recursive(node.left, M_pred, M_gt, T_mask, num_vars, N, tau, pre_data)
            right = self._evaluate_recursive(node.right, M_pred, M_gt, T_mask, num_vars, N, tau, pre_data)
            if node.op == '&':
                # Lukasiewicz t-norm: max(0, L + R - 1)
                return torch.clamp(left + right - 1.0, min=0.0, max=1.0)
            elif node.op == '|':
                # Lukasiewicz t-conorm: min(1, L + R)
                return torch.clamp(left + right, min=0.0, max=1.0)
            elif node.op == '=>': 
                # Lukasiewicz implication: min(1, 1 - L + R)
                return torch.clamp(1.0 - left + right, min=0.0, max=1.0)
            elif node.op == '<=>': 
                # Smooth IFF: 1 - |L - R|
                # |x| = max(x, -x) = softplus(x) + softplus(-x) ? No.
                # Smooth abs: sqrt(x^2 + eps) or softplus(x) + softplus(-x)?
                # Let's use sqrt approx or derived from AND/OR.
                # A <=> B == (A => B) & (B => A)
                # Let's keep it simple or usage is rare.
                # User specifically asked for non-ReLU.
                # 1 - abs(L-R).
                # Smooth abs: (L-R)*tanh((L-R)/tau)?
                # Or simply keep torch.abs? It has subgradient.
                return 1.0 - torch.abs(left - right)
        return torch.tensor(0.0, device=M_pred.device)

    def _precompute_batch_rules(self, M_gt, T_mask, ctx: BatchedContext) -> Dict[int, PrecomputedRuleData]:
        batch_pre_data = {}
        N = ctx.max_ent
        for rule in self.compiled_rules:
            accum_mask = None
            for v_idx in range(rule.num_vars):
                m_broad = self._broadcast_term(ctx.entity_mask, [v_idx], rule.num_vars, N)
                accum_mask = m_broad if accum_mask is None else accum_mask * m_broad
            for atom in rule.rel_atoms:
                p_mask_broad = self._broadcast_term(ctx.pair_mask, atom.spec.var_indices, rule.num_vars, N)
                accum_mask = p_mask_broad if accum_mask is None else accum_mask * p_mask_broad
            constant_atoms = {}
            def collect_constants(node):
                if isinstance(node, CompiledAtom):
                    spec = node.spec
                    if spec.source_tensor == 'T' and T_mask is not None:
                        val = T_mask[..., spec.tensor_idx]
                        broad = self._broadcast_term(val, spec.var_indices, rule.num_vars, N)
                        if spec.is_negated: broad = 1.0 - broad
                        constant_atoms[id(node)] = broad
                    elif spec.source_tensor == 'M_GT' and M_gt is not None:
                        val = M_gt[..., spec.tensor_idx]
                        broad = self._broadcast_term(val, spec.var_indices, rule.num_vars, N)
                        if spec.is_negated: broad = 1.0 - broad
                        constant_atoms[id(node)] = broad
                elif isinstance(node, CompiledUnary): collect_constants(node.child)
                elif isinstance(node, CompiledBinary): collect_constants(node.left); collect_constants(node.right)
            collect_constants(rule.root)
            batch_pre_data[rule.rule_idx] = PrecomputedRuleData(accum_mask=accum_mask, constant_atoms=constant_atoms)
        return batch_pre_data

    def _precompute_batch_rules_subset(self, rules, M_gt, T_mask, ctx: BatchedContext) -> Dict[int, PrecomputedRuleData]:
        batch_pre_data = {}
        N = ctx.max_ent
        for rule in rules:
            accum_mask = None
            for v_idx in range(rule.num_vars):
                m_broad = self._broadcast_term(ctx.entity_mask, [v_idx], rule.num_vars, N)
                accum_mask = m_broad if accum_mask is None else accum_mask * m_broad
            for atom in rule.rel_atoms:
                p_mask_broad = self._broadcast_term(ctx.pair_mask, atom.spec.var_indices, rule.num_vars, N)
                accum_mask = p_mask_broad if accum_mask is None else accum_mask * p_mask_broad
            constant_atoms = {}
            def collect_constants(node):
                if isinstance(node, CompiledAtom):
                    spec = node.spec
                    if spec.source_tensor == 'T' and T_mask is not None:
                        val = T_mask[..., spec.tensor_idx]
                        broad = self._broadcast_term(val, spec.var_indices, rule.num_vars, N)
                        if spec.is_negated: broad = 1.0 - broad
                        constant_atoms[id(node)] = broad
                    elif spec.source_tensor == 'M_GT' and M_gt is not None:
                        val = M_gt[..., spec.tensor_idx]
                        broad = self._broadcast_term(val, spec.var_indices, rule.num_vars, N)
                        if spec.is_negated: broad = 1.0 - broad
                        constant_atoms[id(node)] = broad
                elif isinstance(node, CompiledUnary):
                    collect_constants(node.child)
                elif isinstance(node, CompiledBinary):
                    collect_constants(node.left)
                    collect_constants(node.right)
            collect_constants(rule.root)
            batch_pre_data[rule.rule_idx] = PrecomputedRuleData(accum_mask=accum_mask, constant_atoms=constant_atoms)
        return batch_pre_data

    def _compute_rule_energy(self, Q: torch.Tensor, ctx: BatchedContext, pre_batch_data: Optional[Dict[int, PrecomputedRuleData]] = None, w_logits: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        device = Q.device
        total_energy = torch.tensor(0.0, device=device)
        total_groundings, total_violations = 0.0, 0.0
        rule_details = {}
        if ctx.max_ent == 0 or len(self.compiled_rules) == 0:
            return {"E_rules": total_energy, "total_groundings": 0.0, "violation_count": 0.0, "rule_details": {}}
        
        # Use provided logits (for grad tracking) or self.w_rules_logits
        logits = w_logits if w_logits is not None else self.w_rules_logits
        w_rules = F.softplus(logits)
        hard_mask = self.hard_rule_mask.to(device)
        rule_weights = w_rules * (1.0 - hard_mask) + 1.0 * hard_mask
        B, N, R = ctx.batch_size, ctx.max_ent, self.num_relations
        M_pred = torch.zeros(B, N, N, R, device=device)
        b_idx, r_idx, c_idx = ctx.scatter_indices
        if len(b_idx) > 0:
            M_pred[b_idx, r_idx, c_idx, :] = Q[ctx.global_indices, :]
            inv_idx = torch.tensor(self.inverse_rel_map, device=device)
            M_pred[b_idx, c_idx, r_idx, :] = Q[ctx.global_indices, :][:, inv_idx]
        M_gt, T_mask = None, ctx.T_mask

        def _flatten_conjunction(node):
            if isinstance(node, CompiledBinary) and node.op == '&':
                return _flatten_conjunction(node.left) + _flatten_conjunction(node.right)
            return [node]

        def _compute_implication_violation(rule, pre_data):
            body_nodes = _flatten_conjunction(rule.root.left)
            head_val = self._evaluate_recursive(rule.root.right, M_pred, M_gt, T_mask, rule.num_vars, N, self.smooth_tau, pre_data)
            sum_ai = torch.zeros_like(head_val)
            for node in body_nodes:
                sum_ai = sum_ai + self._evaluate_recursive(node, M_pred, M_gt, T_mask, rule.num_vars, N, self.smooth_tau, pre_data)
            k = len(body_nodes)
            ell = sum_ai - head_val - float(k - 1)
            tau = float(self.smooth_tau)
            if tau > 0:
                violation = tau * F.softplus(ell / tau)
            else:
                violation = F.relu(ell)
            ell_tol = 0.0
            accum_mask = pre_data.accum_mask if pre_data is not None else None
            if accum_mask is None:
                for v_idx in range(rule.num_vars):
                    m_broad = self._broadcast_term(ctx.entity_mask, [v_idx], rule.num_vars, N)
                    accum_mask = m_broad if accum_mask is None else accum_mask * m_broad
                for atom in rule.rel_atoms:
                    p_mask_broad = self._broadcast_term(ctx.pair_mask, atom.spec.var_indices, rule.num_vars, N)
                    accum_mask = p_mask_broad if accum_mask is None else accum_mask * p_mask_broad
            if accum_mask is not None:
                violation = violation * accum_mask
                grounding_count = accum_mask.sum()
                v_count = ((ell > ell_tol) & (accum_mask > 0)).float().sum().item()
            else:
                grounding_count = torch.tensor(float(violation.numel()), device=device)
                v_count = (ell > ell_tol).float().sum().item()
            return violation, grounding_count, v_count
        if pre_batch_data is None and self.rule_chunk_size:
            rules = self.compiled_rules
            for i in range(0, len(rules), self.rule_chunk_size):
                chunk = rules[i:i + self.rule_chunk_size]
                chunk_pre = self._precompute_batch_rules_subset(chunk, M_gt, T_mask, ctx)
                for rule in chunk:
                    pre_data = chunk_pre.get(rule.rule_idx)
                    if isinstance(rule.root, CompiledBinary) and rule.root.op == '=>':
                        violation, grounding_count, v_count = _compute_implication_violation(rule, pre_data)
                        rule_energy = rule_weights[rule.rule_idx] * violation.sum()
                        total_energy = total_energy + rule_energy
                        total_violations += v_count
                        total_groundings += grounding_count.item()
                        rule_details[rule.name] = {
                            'weight': rule_weights[rule.rule_idx].item(),
                            'grounding_count': grounding_count.item(),
                            'violation_count': v_count
                        }
                        continue
                    truth_val = self._evaluate_recursive(rule.root, M_pred, M_gt, T_mask, rule.num_vars, N, self.smooth_tau, pre_data)
                    accum_mask = pre_data.accum_mask if pre_data is not None else None
                    if accum_mask is None:
                        for v_idx in range(rule.num_vars):
                            m_broad = self._broadcast_term(ctx.entity_mask, [v_idx], rule.num_vars, N)
                            accum_mask = m_broad if accum_mask is None else accum_mask * m_broad
                        for atom in rule.rel_atoms:
                             p_mask_broad = self._broadcast_term(ctx.pair_mask, atom.spec.var_indices, rule.num_vars, N)
                             accum_mask = p_mask_broad if accum_mask is None else accum_mask * p_mask_broad
                    if accum_mask is not None:
                        truth_val = truth_val * accum_mask
                        grounding_count = accum_mask.sum()
                    else:
                        grounding_count = torch.tensor(float(truth_val.numel()), device=device)
                    violation = 1.0 - truth_val
                    if accum_mask is not None:
                        violation = violation * accum_mask
                    rule_energy = rule_weights[rule.rule_idx] * violation.sum()
                    total_energy = total_energy + rule_energy
                    violation_tol = 1e-9
                    v_count = (violation > violation_tol).float().sum().item()
                    total_violations += v_count
                    total_groundings += grounding_count.item()
                    rule_details[rule.name] = {'weight': rule_weights[rule.rule_idx].item(), 'grounding_count': grounding_count.item(), 'violation_count': v_count}
                del chunk_pre
        else:
            for rule in self.compiled_rules:
                pre_data = pre_batch_data[rule.rule_idx] if pre_batch_data is not None else None
                if isinstance(rule.root, CompiledBinary) and rule.root.op == '=>':
                    violation, grounding_count, v_count = _compute_implication_violation(rule, pre_data)
                    rule_energy = rule_weights[rule.rule_idx] * violation.sum()
                    total_energy = total_energy + rule_energy
                    total_violations += v_count
                    total_groundings += grounding_count.item()
                    rule_details[rule.name] = {
                        'weight': rule_weights[rule.rule_idx].item(),
                        'grounding_count': grounding_count.item(),
                        'violation_count': v_count
                    }
                    continue
                truth_val = self._evaluate_recursive(rule.root, M_pred, M_gt, T_mask, rule.num_vars, N, self.smooth_tau, pre_data)
                accum_mask = pre_data.accum_mask if pre_data is not None else None
                if accum_mask is None:
                    for v_idx in range(rule.num_vars):
                        m_broad = self._broadcast_term(ctx.entity_mask, [v_idx], rule.num_vars, N)
                        accum_mask = m_broad if accum_mask is None else accum_mask * m_broad
                    for atom in rule.rel_atoms:
                         p_mask_broad = self._broadcast_term(ctx.pair_mask, atom.spec.var_indices, rule.num_vars, N)
                         accum_mask = p_mask_broad if accum_mask is None else accum_mask * p_mask_broad
                if accum_mask is not None:
                    truth_val = truth_val * accum_mask
                    grounding_count = accum_mask.sum()
                else:
                    grounding_count = torch.tensor(float(truth_val.numel()), device=device)
                violation = 1.0 - truth_val
                if accum_mask is not None: violation = violation * accum_mask
                rule_energy = rule_weights[rule.rule_idx] * violation.sum()
                total_energy = total_energy + rule_energy
                violation_tol = 1e-9
                v_count = (violation > violation_tol).float().sum().item()
                total_violations += v_count
                total_groundings += grounding_count.item()
                rule_details[rule.name] = {'weight': rule_weights[rule.rule_idx].item(), 'grounding_count': grounding_count.item(), 'violation_count': v_count}
        return {"E_rules": total_energy, "total_groundings": total_groundings, "violation_count": total_violations, "rule_details": rule_details}

    def _build_doc_contexts(self, Q, entity_pairs, doc_ids, e1_types, e2_types):
        device = Q.device
        doc_to_indices = {}
        for idx, d in enumerate(doc_ids):
            if d not in doc_to_indices: doc_to_indices[d] = []
            doc_to_indices[d].append(idx)
        unique_docs = list(doc_to_indices.keys())
        batch_size = len(unique_docs)
        doc_to_batch_idx = {d: i for i, d in enumerate(unique_docs)}
        doc_ent_to_idx, max_ent, doc_ent_types = {}, 0, {}
        for d_key in unique_docs:
            idxs = doc_to_indices[d_key]
            ent_type_map = {}
            if e1_types is not None:
                for g_idx in idxs:
                    e1, e2 = entity_pairs[g_idx]
                    ent_type_map[e1] = int(e1_types[g_idx].item())
                    ent_type_map[e2] = int(e2_types[g_idx].item())
            doc_ent_types[d_key] = ent_type_map
            entities = set(ent_type_map.keys())
            if not entities:
                for g_idx in idxs:
                    entities.add(entity_pairs[g_idx][0]); entities.add(entity_pairs[g_idx][1])
            sorted_ents = sorted(list(entities))
            doc_ent_to_idx[d_key] = {e: i for i, e in enumerate(sorted_ents)}
            max_ent = max(max_ent, len(sorted_ents))
        if max_ent == 0:
            return BatchedContext(scatter_indices=(torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)), global_indices=torch.empty(0, dtype=torch.long, device=device), T_mask=None, entity_mask=torch.zeros(batch_size, 0, device=device), pair_mask=torch.zeros(batch_size, 0, 0, device=device), batch_size=batch_size, max_ent=0)
        batch_indices_list, row_indices_list, col_indices_list, global_indices_list = [], [], [], []
        entity_mask = torch.zeros(batch_size, max_ent, device=device)
        pair_mask = torch.zeros(batch_size, max_ent, max_ent, device=device)
        T_mask = torch.zeros(batch_size, max_ent, self.num_types, device=device) if e1_types is not None else None
        R = self.num_relations
        for d_key in unique_docs:
            b_idx, local_ent_map = doc_to_batch_idx[d_key], doc_ent_to_idx[d_key]
            entity_mask[b_idx, :len(local_ent_map)] = 1.0
            if T_mask is not None:
                for e, t_id in doc_ent_types[d_key].items():
                    if e in local_ent_map and 0 <= t_id < self.num_types: T_mask[b_idx, local_ent_map[e], t_id] = 1.0
            for g_idx in doc_to_indices[d_key]:
                e1, e2 = entity_pairs[g_idx]
                if e1 in local_ent_map and e2 in local_ent_map:
                    r, c = local_ent_map[e1], local_ent_map[e2]
                    batch_indices_list.append(b_idx); row_indices_list.append(r); col_indices_list.append(c); global_indices_list.append(g_idx)
                    pair_mask[b_idx, r, c] = 1.0
        return BatchedContext(scatter_indices=(torch.tensor(batch_indices_list, device=device, dtype=torch.long), torch.tensor(row_indices_list, device=device, dtype=torch.long), torch.tensor(col_indices_list, device=device, dtype=torch.long)), global_indices=torch.tensor(global_indices_list, device=device, dtype=torch.long), T_mask=T_mask, entity_mask=entity_mask, pair_mask=pair_mask, batch_size=batch_size, max_ent=max_ent)

    def _compute_reason_energy(self, P, Q, ctx, pre_batch_data=None, w_logits=None):
        eps = 1e-12
        rule_result = self._compute_rule_energy(Q, ctx, pre_batch_data, w_logits=w_logits)
        E_rules = rule_result['E_rules']
        E_ent = -(Q * torch.log(Q + eps)).sum(dim=-1).mean()
        E_kl = (Q * (torch.log(Q + eps) - torch.log(P + eps))).sum(dim=-1).mean()
        E_re = E_rules + self.lambda_kl * E_kl + self.lambda_entropy * E_ent
        return {'E_re': E_re,
                'E_rules': E_rules,
                'E_ent': E_ent,
                'E_kl': E_kl,
                'GR': rule_result['total_groundings'],
                'VC': rule_result['violation_count'],
                'rule_details': rule_result['rule_details']}
    
    

    def build_context(self, Q, entity_pairs, doc_ids, e1_types=None, e2_types=None):
        return self._build_doc_contexts(Q, entity_pairs, doc_ids, e1_types, e2_types)

    def _prepare_context(self, Q, entity_pairs, doc_ids, e1_types, e2_types, ctx=None):
        if ctx is None:
            ctx = self.build_context(Q, entity_pairs, doc_ids, e1_types, e2_types)
        pre_batch_data = None if self.rule_chunk_size else self._precompute_batch_rules(None, ctx.T_mask, ctx)
        return ctx, pre_batch_data


    def _deq_forward(self, P, entity_pairs, doc_ids, e1_types, e2_types, ctx, pre_batch_data):
        Q = ImplicitReasoningFunctionV2.apply(
            P, self.w_rules_logits, self, entity_pairs, doc_ids, e1_types, e2_types, ctx
        )
        
        energy_history = []
        with torch.no_grad():
            res_p = self._compute_reason_energy(P, P, ctx, pre_batch_data)
            res_q = self._compute_reason_energy(P, Q, ctx, pre_batch_data)
            energy_history.append({
                    'step': 0,
                    'E_re': res_p['E_re'].item(),
                    'E_rules': res_p['E_rules'].item(),
                    'E_ent': res_p['E_ent'].item(),
                    'E_kl': res_p['E_kl'].item(),
                    'GR': self._to_cpu_scalar(res_p['GR']),
                    'VC': self._to_cpu_scalar(res_p['VC']),
                    'rule_details': res_p['rule_details']
                })
            energy_history.append({
                    'step': self.max_steps,
                    'E_re': res_q['E_re'].item(),
                    'E_rules': res_q['E_rules'].item(),
                    'E_ent': res_q['E_ent'].item(),
                    'E_kl': res_q['E_kl'].item(),
                    'GR': self._to_cpu_scalar(res_q['GR']),
                    'VC': self._to_cpu_scalar(res_q['VC']),
                    'rule_details': res_q['rule_details']
                })
        return Q, energy_history

    def _unrolled_forward(self, P, ctx, pre_batch_data):
        Q = P.clone()
        energy_history = []
        eps, is_train = 1e-12, self.training
        
        old_E_re = float('inf')
        
        last_step = 0
        
        if is_train:
            for step in range(self.max_steps):
                last_step = step
                res = self._compute_reason_energy(P, Q, ctx, pre_batch_data)
                E_re = res['E_re']

                energy_history.append({
                    'step': step,
                    'E_re': res['E_re'].item(),
                    'E_rules': res['E_rules'].item(),
                    'E_ent': res['E_ent'].item(),
                    'E_kl': res['E_kl'].item(),
                    'GR': self._to_cpu_scalar(res['GR']),
                    'VC': self._to_cpu_scalar(res['VC']),
                    'rule_details': res['rule_details']
                })

                g_q, = torch.autograd.grad(E_re, Q, create_graph=True)
                g_q = torch.clamp(g_q, min=-self.grad_clip_value, max=self.grad_clip_value)
                log_Q = torch.log(Q + eps) - self.step_size * g_q
                Q = torch.softmax(log_Q, dim=-1)
                if abs(E_re.item() - old_E_re) < self.tol:
                    break
                else:
                    old_E_re = E_re.item()
        else:
            for step in range(self.max_steps):
                last_step = step
                with torch.enable_grad():
                    Q = Q.detach().requires_grad_(True)
                    res = self._compute_reason_energy(P, Q, ctx, pre_batch_data)
                    E_re = res['E_re']
                    energy_history.append({
                        'step': step,
                        'E_re': res['E_re'].item(),
                        'E_rules': res['E_rules'].item(),
                        'E_ent': res['E_ent'].item(),
                        'E_kl': res['E_kl'].item(),
                        'GR': self._to_cpu_scalar(res['GR']),
                        'VC': self._to_cpu_scalar(res['VC']),
                        'rule_details': res['rule_details']
                    })
                    g_q, = torch.autograd.grad(E_re, Q, create_graph=False)
                    g_q = torch.clamp(g_q, min=-self.grad_clip_value, max=self.grad_clip_value)
                    log_Q = torch.log(Q + eps) - self.step_size * g_q
                    Q = torch.softmax(log_Q, dim=-1).detach()
                    if abs(E_re.item() - old_E_re) < self.tol:
                        break
                    else:
                        old_E_re = E_re.item()
                    
        last_res = self._compute_reason_energy(P, Q, ctx, pre_batch_data)
        energy_history.append({
            'step': last_step,
            'E_re': last_res['E_re'].item(),
            'E_rules': last_res['E_rules'].item(),
            'E_ent': last_res['E_ent'].item(),
            'E_kl': last_res['E_kl'].item(),
            'GR': self._to_cpu_scalar(last_res['GR']),
            'VC': self._to_cpu_scalar(last_res['VC']),
            'rule_details': last_res['rule_details']
        })
        
        return Q, energy_history

    def forward(self, P, entity_pairs, doc_ids, e1_types=None, e2_types=None, ctx=None):
        ctx, pre_batch_data = self._prepare_context(P, entity_pairs, doc_ids, e1_types, e2_types, ctx=ctx)
        if self.use_deq:
            return self._deq_forward(P, entity_pairs, doc_ids, e1_types, e2_types, ctx, pre_batch_data)
        return self._unrolled_forward(P, ctx, pre_batch_data)
