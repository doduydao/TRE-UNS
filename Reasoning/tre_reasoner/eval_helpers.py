import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict


def analyze_consistency(doc_predictions, dataset_name='MATRES', vague_id=3):
    """
    Computes Transitivity and Symmetry violation rates.
    doc_predictions: dict[doc_id][(u, v)] = label_id
    """
    if dataset_name == 'MATRES':
        transitivity_table = {
            (0, 0): 0,
            (1, 1): 1,
            (2, 2): 2,
            (0, 2): 0,
            (2, 0): 0,
            (1, 2): 1,
            (2, 1): 1,
        }
        symmetry_map = {0: 1, 1: 0, 2: 2, 3: 3}
    elif dataset_name == 'I2B2':
        transitivity_table = {
            (0, 0): 0,
            (1, 1): 1,
            (2, 2): 2,
            (0, 2): 0,
            (2, 0): 0,
            (1, 2): 1,
            (2, 1): 1,
        }
        symmetry_map = {0: 1, 1: 0, 2: 2}
    else:
        print(f"Warning: Consistency logic not defined for {dataset_name}. Skipping specifics.")
        return 0, 0, 0, 0

    total_triplets = 0
    consistent_triplets = 0
    total_pairs_sym = 0
    consistent_pairs_sym = 0

    for _, relations in doc_predictions.items():
        visited_pairs = set()
        for (u, v), r in relations.items():
            if (u, v) in visited_pairs:
                continue
            if (v, u) in relations:
                visited_pairs.add((v, u))
                total_pairs_sym += 1
                r_inv = relations[(v, u)]
                if symmetry_map.get(r) == r_inv:
                    consistent_pairs_sym += 1

        adj = defaultdict(dict)
        for (u, v), r in relations.items():
            adj[u][v] = r

        nodes = list(adj.keys())
        for u in nodes:
            for v in adj[u]:
                r1 = adj[u][v]
                if vague_id is not None and r1 == vague_id:
                    continue

                if v in adj:
                    for w in adj[v]:
                        if w == u:
                            continue
                        r2 = adj[v][w]
                        if vague_id is not None and r2 == vague_id:
                            continue

                        if w in adj[u]:
                            r3_pred = adj[u][w]
                            if vague_id is not None and r3_pred == vague_id:
                                continue

                            if (r1, r2) in transitivity_table:
                                total_triplets += 1
                                r3_expected = transitivity_table[(r1, r2)]
                                if r3_pred == r3_expected:
                                    consistent_triplets += 1

    tvr = 1.0 - (consistent_triplets / total_triplets) if total_triplets > 0 else 0.0
    svr = 1.0 - (consistent_pairs_sym / total_pairs_sym) if total_pairs_sym > 0 else 0.0

    return tvr, svr, total_triplets, total_pairs_sym


def calculate_gt_energy(model, dataloader, device, dataset_cfg, debug=False, max_samples=3, max_triplets=10, energy_doc_id=None):
    if not hasattr(model, 'reasoning'):
        print("Model has no reasoning layer. Skipping energy.")
        return 0.0

    model.eval()
    total_energy = 0.0
    total_e_rules = 0.0
    total_e_kl = 0.0
    total_batches = 0
    rule_stats = defaultdict(lambda: {
        'weight_sum': 0.0,
        'violations': 0.0,
        'groundings': 0.0,
        'count': 0
    })
    samples_printed = 0
    triplets_printed = 0
    triplets_total = 0
    full_doc_logged = False
    label_map = dataset_cfg.relation_map or {}
    id2label = {v: k for k, v in label_map.items()} if label_map else None
    if dataset_cfg.name == 'MATRES':
        transitivity_table = {
            (0, 0): 0,
            (1, 1): 1,
            (2, 2): 2,
            (0, 2): 0,
            (2, 0): 0,
            (1, 2): 1,
            (2, 1): 1
        }
    elif dataset_cfg.name == 'I2B2':
        transitivity_table = {
            (0, 0): 0,
            (1, 1): 1,
            (2, 2): 2,
            (0, 2): 0,
            (2, 0): 0,
            (1, 2): 1,
            (2, 1): 1
        }
    else:
        transitivity_table = {}

    def _log_all_rule_groundings_for_doc(labels, entity_pairs, doc_id):
        if not entity_pairs:
            print(f"Doc {doc_id}: No entity pairs found for logging.")
            return
        rel_pairs = defaultdict(dict)
        for (e1, e2), lbl in zip(entity_pairs, labels.tolist()):
            rel_pairs[e1][e2] = lbl

        entities = sorted({e for pair in entity_pairs for e in pair})
        ent_to_idx = {e: i for i, e in enumerate(entities)}
        idx_to_ent = {i: e for e, i in ent_to_idx.items()}
        n_ent = len(entities)
        if n_ent == 0:
            print(f"Doc {doc_id}: No entities found for logging.")
            return

        num_rel = model.reasoning.num_relations
        q_onehot = nn.functional.one_hot(labels, num_classes=num_rel).float()
        m_pred = torch.zeros(1, n_ent, n_ent, num_rel, device=labels.device)
        inv_idx = torch.tensor(model.reasoning.inverse_rel_map, device=labels.device)
        for g_idx, (e1, e2) in enumerate(entity_pairs):
            i, j = ent_to_idx[e1], ent_to_idx[e2]
            m_pred[0, i, j, :] = q_onehot[g_idx]
            m_pred[0, j, i, :] = q_onehot[g_idx][inv_idx]

        entity_mask = torch.zeros(1, n_ent, device=labels.device)
        entity_mask[0, :n_ent] = 1.0
        pair_mask = torch.zeros(1, n_ent, n_ent, device=labels.device)
        for e1, e2 in entity_pairs:
            pair_mask[0, ent_to_idx[e1], ent_to_idx[e2]] = 1.0

        violation_tol = 1e-9

        def _flatten_conjunction(node):
            if hasattr(node, 'op') and node.op == '&':
                return _flatten_conjunction(node.left) + _flatten_conjunction(node.right)
            return [node]

        def _lbl(x):
            return id2label.get(x, str(x)) if id2label else str(x)

        def _pair_label(a, b):
            lbl = rel_pairs.get(a, {}).get(b, None)
            return _lbl(lbl) if lbl is not None else "NA"

        print(f"Doc {doc_id}: Full rule grounding log (OK/VIOL)")
        for rule in model.reasoning.compiled_rules:
            if hasattr(rule.root, 'op') and rule.root.op == '=>':
                body_nodes = _flatten_conjunction(rule.root.left)
                head_val = model.reasoning._evaluate_recursive(
                    rule.root.right,
                    m_pred,
                    None,
                    None,
                    rule.num_vars,
                    n_ent,
                    model.reasoning.smooth_tau,
                    None
                )
                sum_ai = torch.zeros_like(head_val)
                for node in body_nodes:
                    sum_ai = sum_ai + model.reasoning._evaluate_recursive(
                        node,
                        m_pred,
                        None,
                        None,
                        rule.num_vars,
                        n_ent,
                        model.reasoning.smooth_tau,
                        None
                    )
                k = len(body_nodes)
                ell = sum_ai - head_val - float(k - 1)
                tau = float(model.reasoning.smooth_tau)
                if tau > 0:
                    viol = tau * F.softplus(ell / tau)
                else:
                    viol = torch.relu(ell)
                truth_val = None
            else:
                truth_val = model.reasoning._evaluate_recursive(
                    rule.root,
                    m_pred,
                    None,
                    None,
                    rule.num_vars,
                    n_ent,
                    model.reasoning.smooth_tau,
                    None
                )
            accum_mask = None
            for v_idx in range(rule.num_vars):
                m_broad = model.reasoning._broadcast_term(entity_mask, [v_idx], rule.num_vars, n_ent)
                accum_mask = m_broad if accum_mask is None else accum_mask * m_broad
            for atom in rule.rel_atoms:
                p_mask_broad = model.reasoning._broadcast_term(pair_mask, atom.spec.var_indices, rule.num_vars, n_ent)
                accum_mask = p_mask_broad if accum_mask is None else accum_mask * p_mask_broad
            if accum_mask is not None:
                if truth_val is not None:
                    truth_val = truth_val * accum_mask
                else:
                    ell = ell * accum_mask
                    viol = viol * accum_mask
                mask_idx = (accum_mask > 0).nonzero(as_tuple=False)
            else:
                if truth_val is not None:
                    mask_idx = torch.ones_like(truth_val, dtype=torch.bool).nonzero(as_tuple=False)
                else:
                    mask_idx = torch.ones_like(ell, dtype=torch.bool).nonzero(as_tuple=False)

            for idx in mask_idx:
                if rule.num_vars == 2:
                    _, i, j = idx.tolist()
                    e1, e2 = idx_to_ent[i], idx_to_ent[j]
                    if truth_val is not None:
                        t = float(truth_val[0, i, j].item())
                        v = 1.0 - t
                        ok = v <= violation_tol
                    else:
                        t = float(ell[0, i, j].item())
                        v = float(viol[0, i, j].item())
                        ok = t <= violation_tol
                    r12 = _pair_label(e1, e2)
                    metric_label = "truth" if truth_val is not None else "ell"
                    print(
                        f"Doc {doc_id}: {rule.name} ({e1},{e2}) rel={r12} "
                        f"{metric_label}={t:.6f} viol={v:.6f} [{'OK' if ok else 'VIOL'}]"
                    )
                elif rule.num_vars == 3:
                    _, i, j, k = idx.tolist()
                    e1, e2, e3 = idx_to_ent[i], idx_to_ent[j], idx_to_ent[k]
                    if truth_val is not None:
                        t = float(truth_val[0, i, j, k].item())
                        v = 1.0 - t
                        ok = v <= violation_tol
                    else:
                        t = float(ell[0, i, j, k].item())
                        v = float(viol[0, i, j, k].item())
                        ok = t <= violation_tol
                    r12 = _pair_label(e1, e2)
                    r23 = _pair_label(e2, e3)
                    r13 = _pair_label(e1, e3)
                    metric_label = "truth" if truth_val is not None else "ell"
                    print(
                        f"Doc {doc_id}: {rule.name} ({e1},{e2},{e3}) rels=({r12},{r23},{r13}) "
                        f"{metric_label}={t:.6f} viol={v:.6f} [{'OK' if ok else 'VIOL'}]"
                    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="GT Energy"):
            labels = batch["labels"].to(device)
            if (labels < 0).any():
                continue

            e1_type_ids = batch["e1_type_ids"].to(device)
            e2_type_ids = batch["e2_type_ids"].to(device)
            e1_ids = batch['e1_ids']
            e2_ids = batch['e2_ids']
            doc_ids = batch["doc_ids"]
            if isinstance(doc_ids, torch.Tensor):
                doc_ids = doc_ids.tolist()
            if isinstance(e1_ids, torch.Tensor):
                e1_ids = e1_ids.tolist()
            if isinstance(e2_ids, torch.Tensor):
                e2_ids = e2_ids.tolist()

            if energy_doc_id is not None:
                match_indices = [i for i, d in enumerate(doc_ids) if str(d) == str(energy_doc_id)]
                if not match_indices:
                    continue
                idx_t = torch.tensor(match_indices, device=labels.device, dtype=torch.long)
                labels = labels.index_select(0, idx_t)
                e1_type_ids = e1_type_ids.index_select(0, idx_t)
                e2_type_ids = e2_type_ids.index_select(0, idx_t)
                doc_ids = [doc_ids[i] for i in match_indices]
                e1_ids = [e1_ids[i] for i in match_indices]
                e2_ids = [e2_ids[i] for i in match_indices]

            entity_pairs = list(zip(e1_ids, e2_ids))

            num_classes = model.reasoning.num_relations
            P_gt = nn.functional.one_hot(labels, num_classes=num_classes).float()

            if hasattr(model.reasoning, "compute_rule_loss_with_probs"):
                energy = model.reasoning.compute_rule_loss_with_probs(
                    probs=P_gt,
                    entity_pairs=entity_pairs,
                    doc_ids=doc_ids,
                    e1_types=e1_type_ids,
                    e2_types=e2_type_ids
                )
                total_energy += energy.item()
            else:
                ctx = model.reasoning.build_context(
                    Q=P_gt,
                    entity_pairs=entity_pairs,
                    doc_ids=doc_ids,
                    e1_types=e1_type_ids,
                    e2_types=e2_type_ids,
                )
                res = model.reasoning._compute_reason_energy(P_gt, P_gt, ctx)
                total_energy += res['E_re'].item()
                total_e_rules += res['E_rules'].item()
                total_e_kl += res['E_kl'].item()
                if debug:
                    if samples_printed < max_samples:
                        top_rules = sorted(
                            res.get('rule_details', {}).items(),
                            key=lambda kv: kv[1].get('violation_count', 0),
                            reverse=True
                        )[:5]
                        if energy_doc_id is not None:
                            print(f"Doc {energy_doc_id}: Top rule violations:")
                        else:
                            uniq_doc_ids = doc_ids
                            if isinstance(uniq_doc_ids, torch.Tensor):
                                uniq_doc_ids = uniq_doc_ids.tolist()
                            if isinstance(uniq_doc_ids, list):
                                uniq_doc_ids = sorted(set(uniq_doc_ids))
                            print(f"Sample docs: {uniq_doc_ids[:5]} | Top rule violations:")
                        for r_name, r_info in top_rules:
                            v = r_info.get('violation_count', 0)
                            g = r_info.get('grounding_count', 0)
                            w = r_info.get('weight', 0.0)
                            print(f"  - {r_name}: violations={v}, groundings={g}, weight={w:.4f}")
                        samples_printed += 1

                    if transitivity_table and (max_triplets is None or max_triplets <= 0 or triplets_printed < max_triplets):
                        def _lbl(x):
                            return id2label.get(x, str(x)) if id2label else str(x)

                        doc_rel_pairs = defaultdict(lambda: defaultdict(dict))
                        for idx, ((e1, e2), lbl) in enumerate(zip(entity_pairs, labels.tolist())):
                            d_id = doc_ids[idx]
                            if isinstance(d_id, torch.Tensor):
                                d_id = d_id.item()
                            if energy_doc_id is not None and str(d_id) != str(energy_doc_id):
                                continue
                            doc_rel_pairs[str(d_id)][e1][e2] = lbl

                        for d_id, rel_pairs in doc_rel_pairs.items():
                            for u in rel_pairs:
                                for v, r1 in rel_pairs[u].items():
                                    if dataset_cfg.vague_label_id is not None and r1 == dataset_cfg.vague_label_id:
                                        continue
                                    if v in rel_pairs:
                                        for w, r2 in rel_pairs[v].items():
                                            if w == u:
                                                continue
                                            if dataset_cfg.vague_label_id is not None and r2 == dataset_cfg.vague_label_id:
                                                continue
                                            if w in rel_pairs[u] and (r1, r2) in transitivity_table:
                                                r3 = rel_pairs[u][w]
                                                if dataset_cfg.vague_label_id is not None and r3 == dataset_cfg.vague_label_id:
                                                    continue
                                                expected = transitivity_table[(r1, r2)]
                                                ok = (r3 == expected)
                                                triplets_total += 1
                                                print(
                                                    f"Doc {d_id}: Triplet ({u},{v},{w}) "
                                                    f"{_lbl(r1)} + {_lbl(r2)} -> expected {_lbl(expected)}, got {_lbl(r3)} "
                                                    f"[{'OK' if ok else 'VIOL'}]"
                                                )
                                                triplets_printed += 1
                                                if max_triplets is not None and max_triplets > 0 and triplets_printed >= max_triplets:
                                                    break
                                        if max_triplets is not None and max_triplets > 0 and triplets_printed >= max_triplets:
                                            break
                                if max_triplets is not None and max_triplets > 0 and triplets_printed >= max_triplets:
                                    break
                            if max_triplets is not None and max_triplets > 0 and triplets_printed >= max_triplets:
                                break
                        if energy_doc_id is not None and triplets_total == 0:
                            print(f"Doc {energy_doc_id}: No transitivity triplets found.")

                    if energy_doc_id is not None and not full_doc_logged:
                        _log_all_rule_groundings_for_doc(labels, entity_pairs, energy_doc_id)
                        full_doc_logged = True
                    for r_name, r_info in res.get('rule_details', {}).items():
                        rule_stats[r_name]['weight_sum'] += float(r_info.get('weight', 0.0))
                        rule_stats[r_name]['violations'] += float(r_info.get('violation_count', 0.0))
                        rule_stats[r_name]['groundings'] += float(r_info.get('grounding_count', 0.0))
                        rule_stats[r_name]['count'] += 1
            total_batches += 1

    avg_energy = total_energy / total_batches if total_batches > 0 else 0.0
    if debug and total_batches > 0 and rule_stats:
        print(f"Avg E_rules: {total_e_rules / total_batches:.6f} | Avg E_kl: {total_e_kl / total_batches:.6f}")
        ranked = []
        for r_name, stats in rule_stats.items():
            c = max(1, stats['count'])
            avg_weight = stats['weight_sum'] / c
            ranked.append((stats['violations'], stats['groundings'], avg_weight, r_name))
        ranked.sort(reverse=True)
        print("Top violating rules (by violations):")
        for v, g, w, name in ranked[:10]:
            print(f"  - {name}: violations={v:.0f}, groundings={g:.0f}, avg_weight={w:.4f}")
    return avg_energy


def calculate_pred_energy(model, dataloader, device, dataset_cfg, debug=False, use_q_as_pred=False):
    if not hasattr(model, 'reasoning'):
        print("Model has no reasoning layer. Skipping energy.")
        return 0.0

    model.eval()
    total_energy = 0.0
    total_e_rules = 0.0
    total_batches = 0
    rule_stats = defaultdict(lambda: {
        'weight_sum': 0.0,
        'violations': 0.0,
        'groundings': 0.0,
        'count': 0
    })

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pred Energy"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_marks = batch['word_marks'].to(device)
            e1_marks = batch['e1_marks'].to(device)
            e2_marks = batch['e2_marks'].to(device)

            doc_ids = batch['doc_ids']
            e1_ids = batch['e1_ids']
            e2_ids = batch['e2_ids']

            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'word_marks': word_marks,
                'e1_marks': e1_marks,
                'e2_marks': e2_marks,
                'entity_pairs': list(zip(e1_ids, e2_ids)),
                'doc_ids': doc_ids,
                'e1_type_ids': batch['e1_type_ids'].to(device),
                'e2_type_ids': batch['e2_type_ids'].to(device)
            }

            out = model(**inputs)
            if isinstance(out, dict):
                logits_logic = out.get('L_F')
                q_probs = out.get('Q_T')
                if use_q_as_pred and q_probs is not None:
                    probs = q_probs
                else:
                    probs = torch.softmax(logits_logic, dim=-1)
            else:
                probs = torch.softmax(out, dim=-1)

            ctx = model.reasoning.build_context(
                Q=probs,
                entity_pairs=list(zip(e1_ids, e2_ids)),
                doc_ids=doc_ids,
                e1_types=batch['e1_type_ids'].to(device),
                e2_types=batch['e2_type_ids'].to(device),
            )
            res = model.reasoning._compute_reason_energy(probs, probs, ctx)
            total_energy += res['E_re'].item()
            total_e_rules += res['E_rules'].item()
            for r_name, r_info in res.get('rule_details', {}).items():
                rule_stats[r_name]['weight_sum'] += float(r_info.get('weight', 0.0))
                rule_stats[r_name]['violations'] += float(r_info.get('violation_count', 0.0))
                rule_stats[r_name]['groundings'] += float(r_info.get('grounding_count', 0.0))
                rule_stats[r_name]['count'] += 1
            total_batches += 1

    avg_energy = total_energy / total_batches if total_batches > 0 else 0.0
    if debug and total_batches > 0 and rule_stats:
        print(f"Avg E_rules: {total_e_rules / total_batches:.6f}")
        ranked = []
        for r_name, stats in rule_stats.items():
            c = max(1, stats['count'])
            avg_weight = stats['weight_sum'] / c
            ranked.append((stats['violations'], stats['groundings'], avg_weight, r_name))
        ranked.sort(reverse=True)
        print("Top violating rules (by violations):")
        for v, g, w, name in ranked[:10]:
            print(f"  - {name}: violations={v:.0f}, groundings={g:.0f}, avg_weight={w:.4f}")
    return avg_energy
