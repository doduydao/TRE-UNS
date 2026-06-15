import argparse
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})


def plot_scalability(csv_path, output_img, title_suffix="TRE-UNS", artifact_dir=None):
    # Load data
    df = pd.read_csv(csv_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

    # Color palette for batch sizes
    batch_sizes = sorted(df['batch_size'].unique())
    palette = sns.color_palette("viridis", n_colors=len(batch_sizes))
    color_map = dict(zip(batch_sizes, palette))

    # Plot 1: Reasoning Time vs Max Steps
    ax1 = axes[0]
    for bs in batch_sizes:
        subset = df[df['batch_size'] == bs].sort_values('max_steps')
        ax1.plot(
            subset['max_steps'], 
            subset['reasoning_time'], 
            marker='o', 
            linewidth=2, 
            color=color_map[bs], 
            label=f"Batch Size {bs}"
        )

    ax1.set_xscale('log')
    ax1.set_xlabel("Max Steps (Log Scale)")
    ax1.set_ylabel("Reasoning Time (seconds)")
    ax1.set_title("Reasoning Execution Time vs Max Steps")
    ax1.set_xticks([50, 100, 500, 1000, 5000])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.legend(title="Configuration")

    # Plot 2: Average Actual Steps vs Max Steps
    ax2 = axes[1]
    for bs in batch_sizes:
        subset = df[df['batch_size'] == bs].sort_values('max_steps')
        ax2.plot(
            subset['max_steps'], 
            subset['avg_actual_steps'], 
            marker='s', 
            linewidth=2, 
            linestyle='--',
            color=color_map[bs], 
            label=f"Batch Size {bs}"
        )

    ax2.set_xscale('log')
    ax2.set_xlabel("Max Steps (Log Scale)")
    ax2.set_ylabel("Avg Actual Steps (before convergence)")
    ax2.set_title("Average Reasoning Steps Executed vs Max Steps")
    ax2.set_xticks([50, 100, 500, 1000, 5000])
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend(title="Configuration")

    plt.suptitle(f"Scalability Analysis of {title_suffix}", weight='bold', y=0.98)
    plt.tight_layout()

    # Save image
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Saved plot locally to {output_img}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scalability benchmark results")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark CSV")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--title", type=str, default="TRE-UNS", help="Title suffix")
    args = parser.parse_args()

    plot_scalability(args.csv, args.output, args.title)
