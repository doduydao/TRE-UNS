import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# --- Paths ---
csv_path = "results/neupsl_scalability_results_TBD.csv"
output_img = "results/neupsl_scalability_charts_tbd.png"
# artifact_dir = "/home/prof/ddao/.gemini/antigravity-ide/brain/b4e9f7c6-2c45-4459-bba8-ff010fcbd3e3"
# artifact_img = os.path.join(artifact_dir, "neupsl_scalability_charts.png")


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

plt.suptitle("Scalability Analysis of NeuPSL on TimeBank-Dense (Test)", weight='bold', y=0.98)
plt.tight_layout()

# Save image
os.makedirs(os.path.dirname(output_img), exist_ok=True)
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Saved plot locally to {output_img}")

# # Copy to artifact directory for display
# if os.path.exists(artifact_dir):
#     shutil.copy(output_img, artifact_img)
#     print(f"Copied plot to artifact path: {artifact_img}")
# else:
#     print("Warning: Artifact directory does not exist.")
