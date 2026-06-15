import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["AFTER", "BEFORE", "INCLUDES", "IS_INCLUDED", "SIMULTANEOUS", "VAGUE"]
COLORS = {
    "AFTER": "#d62728",
    "BEFORE": "#1f77b4",
    "INCLUDES": "#9467bd",
    "IS_INCLUDED": "#ff7f0e",
    "SIMULTANEOUS": "#2ca02c",
    "VAGUE": "#7f7f7f",
}


def parse_gt_map(gt_text: str):
    """Parse GT mapping from string like '(e6,e14)=AFTER,(e14,e15)=IS_INCLUDED'."""
    gt_map = {}
    if not gt_text:
        return gt_map

    # Robust parser: pair contains a comma, so do not split naively by comma.
    # Match chunks of the form: (eX,eY)=LABEL
    pattern = re.compile(r"(\([^)]*\))\s*=\s*([A-Za-z_]+)")
    for m in pattern.finditer(gt_text):
        pair = m.group(1).strip()
        label = m.group(2).strip()
        gt_map[pair] = label

    return gt_map


def infer_pairs(df: pd.DataFrame):
    pairs = []
    for col in df.columns:
        for label in LABELS:
            suffix = f"_{label}"
            if col.endswith(suffix):
                pair = col[: -len(suffix)]
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def plot_single_pair(df: pd.DataFrame, pair: str, out_path: Path, gt_map=None):
    if gt_map is None:
        gt_map = {}
    steps = df["step"].tolist()
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in LABELS:
        col = f"{pair}_{label}"
        if col not in df.columns:
            continue
        y = df[col].to_numpy(dtype=float)
        max_y = float(np.max(y)) if len(y) else 0.0
        # Always show every label, even tiny-probability classes.
        lw = 2.8 if max_y >= 0.25 else 1.9
        alpha = 1.0 if max_y >= 0.25 else 0.9
        ax.plot(x, y, "o-", label=label, color=COLORS.get(label, "#444444"), linewidth=lw, alpha=alpha)

    ax.set_xticks(x)
    ax.set_xticklabels([f"t={s}" for s in steps], rotation=25)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Probability")
    ax.set_title(f"Distribution Shift for {pair}")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_triplet_panel(df: pd.DataFrame, pairs, out_path: Path, gt_map=None):
    if gt_map is None:
        gt_map = {}
    steps = df["step"].tolist()
    x = np.arange(len(steps))

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.8), sharey=True)
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        for label in LABELS:
            col = f"{pair}_{label}"
            if col not in df.columns:
                continue
            y = df[col].to_numpy(dtype=float)
            max_y = float(np.max(y)) if len(y) else 0.0
            # Keep all labels visible in the panel as well.
            lw = 2.8 if max_y >= 0.25 else 1.9
            alpha = 1.0 if max_y >= 0.25 else 0.9
            ax.plot(x, y, "o-", label=label, color=COLORS.get(label, "#444444"), linewidth=lw, alpha=alpha)

        ax.set_xticks(x)
        ax.set_xticklabels([f"t={s}" for s in steps], rotation=25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(pair)
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Probability")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=True)

    if gt_map:
        gt_summary = "; ".join([f"{p}={gt_map[p]}" for p in pairs if p in gt_map])
        fig.suptitle(
            f"Probability Trajectories Across Reasoning Steps | GT: {gt_summary}",
            y=1.12,
            fontsize=12,
            fontweight="bold",
        )
    else:
        fig.suptitle("Probability Trajectories Across Reasoning Steps", y=1.12, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="q_trajectory.csv", help="Path to q_trajectory.csv")
    parser.add_argument("--out-prefix", default="case_study_triplet", help="Output prefix for image files")
    parser.add_argument(
        "--gt",
        default="",
        help="GT mapping, e.g. '(e6,e14)=AFTER,(e14,e15)=IS_INCLUDED,(e6,e15)=AFTER'",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = (csv_path.parent / out_prefix).resolve()

    df = pd.read_csv(csv_path)
    if "step" not in df.columns:
        raise ValueError("Input CSV must contain 'step' column.")

    pairs = infer_pairs(df)
    if len(pairs) < 1:
        raise ValueError("Could not infer any relation pair columns from CSV.")

    gt_map = parse_gt_map(args.gt)

    for pair in pairs:
        clean_name = pair.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
        out_file = out_prefix.parent / f"{out_prefix.name}_{clean_name}.png"
        plot_single_pair(df, pair, out_file, gt_map=gt_map)
        print(f"Saved: {out_file}")

    panel_png = out_prefix.parent / f"{out_prefix.name}_panel.png"
    panel_pdf = out_prefix.parent / f"{out_prefix.name}_panel.pdf"
    plot_triplet_panel(df, pairs, panel_png, gt_map=gt_map)
    plot_triplet_panel(df, pairs, panel_pdf, gt_map=gt_map)
    print(f"Saved: {panel_png}")
    print(f"Saved: {panel_pdf}")


if __name__ == "__main__":
    main()
