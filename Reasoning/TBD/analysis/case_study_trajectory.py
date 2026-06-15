"""
Paper Case Study: Step-by-Step Reasoning Trajectory — TBD dataset (TBD_old)
=============================================================================
Document tập trung: APW19980227.0489
Steps: 0 (encoder), 1, 50, 100, 1000
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

STEP_FILES = {
    0:    BASE / "prediction_steps0.csv",
    1:    BASE / "prediction_steps1_v2.csv",
    50:   BASE / "prediction_steps50_v2.csv",
    100:  BASE / "prediction_steps100_v2.csv",
    1000: BASE / "prediction_steps1000_v2.csv",
}

STEPS = [0, 1, 50, 100, 1000]
STEP_LABELS = ["t=0\n(encoder)", "t=1", "t=50", "t=100", "t=1000"]

FOCUS_DOC = "APW19980227.0489"

ALL_RELS = ["BEFORE", "AFTER", "SIMULTANEOUS", "INCLUDES", "IS_INCLUDED", "VAGUE"]

# Clear transitivity rules
TRANS_TABLE = {
    ("BEFORE",       "BEFORE"):       "BEFORE",
    ("AFTER",        "AFTER"):        "AFTER",
    ("SIMULTANEOUS", "BEFORE"):       "BEFORE",
    ("BEFORE",       "SIMULTANEOUS"): "BEFORE",
    ("SIMULTANEOUS", "AFTER"):        "AFTER",
    ("AFTER",        "SIMULTANEOUS"): "AFTER",
    ("SIMULTANEOUS", "SIMULTANEOUS"): "SIMULTANEOUS",
    ("INCLUDES",     "INCLUDES"):     "INCLUDES",
    ("IS_INCLUDED",  "IS_INCLUDED"):  "IS_INCLUDED",
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_pred_dict(step, doc_id=None):
    df = pd.read_csv(STEP_FILES[step])
    if doc_id:
        df = df[df["doc_id"] == doc_id]
    pred = {(r["e1_id"], r["e2_id"]): r["prediction"]   for _, r in df.iterrows()}
    gt   = {(r["e1_id"], r["e2_id"]): r["ground_truth"] for _, r in df.iterrows()}
    return pred, gt


def find_violations(pred_dict):
    rel_graph = {}
    for (e1, e2), r in pred_dict.items():
        rel_graph.setdefault(r, {}).setdefault(e1, []).append(e2)
    viols = set()
    for (r_ab, r_bc), expected in TRANS_TABLE.items():
        for a, b_list in rel_graph.get(r_ab, {}).items():
            for b in b_list:
                for c in rel_graph.get(r_bc, {}).get(b, []):
                    if a == c:
                        continue
                    r_ac = pred_dict.get((a, c))
                    if r_ac is not None and r_ac != expected and r_ac != "VAGUE":
                        viols.add((a, b, c, r_ab, r_bc, expected, r_ac))
    return viols


def violations_by_step(doc_id, steps=STEPS):
    return {s: find_violations(load_pred_dict(s, doc_id)[0]) for s in steps}


def all_doc_violations(steps=STEPS):
    df0 = pd.read_csv(STEP_FILES[0])
    docs = [d for d in df0["doc_id"].unique() if d != "doc_id"]
    return {doc: violations_by_step(doc, steps) for doc in docs}


# ─────────────────────────────────────────────────────────────
# 1. Violation counts — ALL docs per step
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("TBD Case Study — Transitivity Violation Trajectory")
print("=" * 70)

all_viols = all_doc_violations()
total_per_step = {s: sum(len(all_viols[d][s]) for d in all_viols) for s in STEPS}

print("\n[1] Violation counts across ALL documents:")
print("  " + "  ".join(f"t={s:<6}" for s in STEPS))
print("  " + "  ".join(f"{total_per_step[s]:<8}" for s in STEPS))

print(f"\n  {'Document':<30} " + "  ".join(f"t={s:<4}" for s in STEPS))
print("  " + "-" * 65)
for doc in all_viols:
    counts = [len(all_viols[doc][s]) for s in STEPS]
    print(f"  {doc:<30} " + "  ".join(f"{c:<6}" for c in counts))


# ─────────────────────────────────────────────────────────────
# 2. Focus doc: find best triplet (search FOCUS_DOC first, else all docs)
# ─────────────────────────────────────────────────────────────
def find_best_triplet_for_doc(doc, doc_viols, steps=STEPS):
    pred_dicts = {}
    gt_dict = {}
    for s in steps:
        p, g = load_pred_dict(s, doc)
        pred_dicts[s] = p
        gt_dict.update(g)

    step0_keys = {(v[0], v[1], v[2]): v for v in doc_viols[0]}
    triplets = []
    for (A, B, C), vinfo in step0_keys.items():
        statuses = [(A,B,C) in {(v[0],v[1],v[2]) for v in doc_viols[s]} for s in steps]
        n_ch = sum(1 for i in range(1, len(statuses)) if statuses[i] != statuses[i-1])
        triplets.append({
            "triplet": (A,B,C), "vinfo": vinfo, "statuses": statuses,
            "resolved": not statuses[-1], "n_changes": n_ch,
            "ab": [pred_dicts[s].get((A,B), "—") for s in steps],
            "bc": [pred_dicts[s].get((B,C), "—") for s in steps],
            "ac": [pred_dicts[s].get((A,C), "—") for s in steps],
            "pred_dicts": pred_dicts, "gt_dict": gt_dict,
        })
    triplets.sort(key=lambda x: (-int(x["resolved"]), -x["n_changes"]))
    return triplets


focus_triplets = find_best_triplet_for_doc(FOCUS_DOC, all_viols[FOCUS_DOC])

if focus_triplets:
    best = focus_triplets[0]
    selected_doc = FOCUS_DOC
else:
    # Fallback: find best across all docs
    all_candidates = []
    for doc in all_viols:
        candidates = find_best_triplet_for_doc(doc, all_viols[doc])
        for c in candidates:
            c["doc"] = doc
        all_candidates.extend(candidates)
    all_candidates.sort(key=lambda x: (-int(x["resolved"]), -x["n_changes"]))
    best = all_candidates[0]
    selected_doc = best["doc"]
    print(f"\n  FOCUS_DOC has no step-0 violations. Using: {selected_doc}")

A, B, C = best["triplet"]
r_AB_rule, r_BC_rule, expected_rel = best["vinfo"][3], best["vinfo"][4], best["vinfo"][5]
pred_dicts = best["pred_dicts"]
gt_dict = best["gt_dict"]

print(f"\n[2] Best triplet for case study (doc={selected_doc}):")
print(f"  Triplet  : ({A}, {B}, {C})")
print(f"  Rule     : {r_AB_rule} ∘ {r_BC_rule} → {expected_rel}  (violated by actual AC prediction)")
print(f"  Resolved : {best['resolved']} (violation gone at last step)")

print(f"\n  {'Step':<6} | {f'({A},{B})':<14} | {f'({B},{C})':<14} | {f'({A},{C})':<14} | Violation")
print("  " + "-" * 68)
for i, s in enumerate(STEPS):
    flag = "✗ YES" if best["statuses"][i] else "✓ NO "
    print(f"  {s:<6} | {best['ab'][i]:<14} | {best['bc'][i]:<14} | {best['ac'][i]:<14} | {flag}")

gt_AB = gt_dict.get((A,B), "?")
gt_BC = gt_dict.get((B,C), "?")
gt_AC = gt_dict.get((A,C), "?")
print(f"\n  Ground truth: ({A},{B})={gt_AB}  ({B},{C})={gt_BC}  ({A},{C})={gt_AC}")


# ─────────────────────────────────────────────────────────────
# 3. Plot: 2 panels
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle(
    f"TBD — Transitivity Violation Trajectory  |  Doc: {selected_doc}",
    fontsize=12, fontweight="bold", y=1.01)

x = np.arange(len(STEPS))

# ── Panel A: violation counts ──
ax1 = axes[0]
totals = [total_per_step[s] for s in STEPS]

# Stacked bar per document
doc_list = [d for d in all_viols if max(len(all_viols[d][s]) for s in STEPS) > 0]
palette = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22",
           "#16a085", "#d35400", "#7f8c8d", "#c0392b"]
bottom = np.zeros(len(STEPS))
for di, doc in enumerate(doc_list):
    counts = np.array([len(all_viols[doc][s]) for s in STEPS], dtype=float)
    col = palette[di % len(palette)]
    short_name = doc.split(".")[0]
    ax1.bar(x, counts, bottom=bottom, color=col, alpha=0.75,
            label=short_name, width=0.45)
    bottom += counts

# Total label on top
for xi, v in enumerate(totals):
    if v > 0:
        ax1.annotate(str(v), (xi, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=11, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(STEP_LABELS, fontsize=9)
ax1.set_ylabel("# transitivity violations", fontsize=10)
ax1.set_title("(a) Violations per reasoning step\n(all documents)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.7)
ax1.set_ylim(0, max(totals) * 1.7 + 1)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel B: prediction trajectory ──
ax2 = axes[1]
rel_ranks = {r: i for i, r in enumerate(ALL_RELS)}

style = [
    ((A, B), best["ab"], "solid",   "#2980b9"),
    ((B, C), best["bc"], "dashed",  "#e67e22"),
    ((A, C), best["ac"], "dashdot", "#e74c3c"),
]

for (pair, preds, ls, col) in style:
    y_nums = [rel_ranks.get(p, -1) for p in preds]
    gt_val = gt_dict.get(pair, "?")
    ax2.plot(x, y_nums, marker="o", linestyle=ls, linewidth=2.5,
             markersize=9, color=col,
             label=f"({pair[0]},{pair[1]})  [GT: {gt_val}]")
    for xi, (yn, p) in enumerate(zip(y_nums, preds)):
        if yn >= 0:
            short = "IS_I" if p == "IS_INCLUDED" else p[:3]
            ax2.annotate(short, (xi, yn), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=col,
                        fontweight="bold")

for xi, s in enumerate(STEPS):
    if best["statuses"][xi]:
        ax2.axvspan(xi - 0.38, xi + 0.38, alpha=0.09, color="red", zorder=0)

ax2.set_xticks(x)
ax2.set_xticklabels(STEP_LABELS, fontsize=9)
ax2.set_yticks(range(len(ALL_RELS)))
ax2.set_yticklabels(ALL_RELS, fontsize=9)
ax2.set_title(
    f"(b) Prediction trajectory — ({A}, {B}, {C})\n"
    f"Violated rule: {r_AB_rule} ∘ {r_BC_rule} → {expected_rel}",
    fontsize=11, fontweight="bold")
ax2.grid(alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

viol_patch = mpatches.Patch(color="red", alpha=0.15, label="Step with violation (shaded)")
h, l = ax2.get_legend_handles_labels()
ax2.legend(handles=h + [viol_patch], fontsize=8.5, loc="best")

plt.tight_layout(pad=2.5)
out_png = Path(__file__).parent / "case_study_trajectory.png"
out_pdf = Path(__file__).parent / "case_study_trajectory.pdf"
plt.savefig(out_png, bbox_inches="tight", dpi=180)
plt.savefig(out_pdf, bbox_inches="tight")
print(f"\n[3] Plot saved:\n  {out_png}\n  {out_pdf}")


# ─────────────────────────────────────────────────────────────
# 4. LaTeX table
# ─────────────────────────────────────────────────────────────
lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    (r"\caption{Case study: reasoning trajectory for a transitivity violation "
     rf"({r_AB_rule} $\circ$ {r_BC_rule} $\Rightarrow$ {expected_rel}) "
     rf"in document \texttt{{{selected_doc}}}. "
     r"Shaded cells mark the inconsistent conclusion pair.}"),
    r"\label{tab:case-study-tbd}",
    r"\begin{tabular}{lccccc}",
    r"\toprule",
    (r"\textbf{Relation pair} & \textbf{t=0 (enc.)} & \textbf{t=1} & "
     r"\textbf{t=50} & \textbf{t=100} & \textbf{t=1000} \\"),
    r"\midrule",
]

for (pair, preds, _, _) in style:
    gt_val = gt_dict.get(pair, "?")
    cells = []
    for ps, s in zip(preds, STEPS):
        si = STEPS.index(s)
        is_vp = (pair == (A,C)) and best["statuses"][si]
        cells.append(r"\cellcolor{red!12}\textit{" + ps + "}" if is_vp else ps)
    lines.append(
        f"({pair[0]}, {pair[1]}) [GT: {gt_val}] & " +
        " & ".join(cells) + r" \\"
    )

lines.append(r"\midrule")
viol_row = []
for s in STEPS:
    viol_row.append(
        r"\textcolor{red}{$\times$}" if best["statuses"][STEPS.index(s)]
        else r"\textcolor{teal}{$\checkmark$}"
    )
lines.append(r"Violation & " + " & ".join(viol_row) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

print("\n[4] LaTeX TABLE:")
print("=" * 70)
for l in lines:
    print(l)

tex_path = Path(__file__).parent / "case_study_table.tex"
with open(tex_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nLaTeX saved to: {tex_path}")
