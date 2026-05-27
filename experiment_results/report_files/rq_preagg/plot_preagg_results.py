import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from pathlib import Path

FANOUT = [20, 10]

dir = Path(__file__).resolve().parent
batch_sizes = [32, 64, 128, 256, 512]

def build_df(dir: Path, batch_sizes: list[int], fanout: list[int]) -> pd.DataFrame:
    techniques = ["no_preagg", "preagg"]
    datasets = ["products", "coauthor"]
    dataset_displayname = {
        "products": "obgn-products",
        "coauthor": "coauthor"
    }
    data = []

    def parse_subdir(dir, dataset, batch_size, techniques, fanout) -> dict:
        fandir = "_".join(map(str, fanout))
        baseline = json.loads(
            (dir / techniques[0] / dataset / fandir / str(batch_size) / "measurements.json").read_text(encoding="utf-8")
        )
        improvement = json.loads(
            (dir / techniques[1] / dataset / fandir / str(batch_size) / "measurements.json").read_text(encoding="utf-8")
        )

        a = baseline["metrics"]["avg_feat_bytes"]
        b = improvement["metrics"]["avg_feat_bytes"]
        c = baseline["metrics"]["remote_feature_latency_s"]["mean_s"]
        d = improvement["metrics"]["remote_feature_latency_s"]["mean_s"]

        return {
            "Batch Size": batch_size,
            "Bytes Fraction": b / a,
            "Relative Mean Latency": d / c,
            "Dataset": dataset_displayname[dataset]
        }



    for ds in datasets:
        for batch_size in batch_sizes:
            data.append(parse_subdir(dir, ds, batch_size, techniques, fanout))

    return pd.DataFrame(data)

df = pd.DataFrame(build_df(dir, batch_sizes, fanout=FANOUT))

# Set theme for a more professional look
sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
plt.rcParams.update({
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
    'figure.dpi': 300
})

# Create the side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

sns.lineplot(
    data=df, 
    x="Batch Size", 
    y="Bytes Fraction", 
    hue="Dataset", 
    marker="o",
    markersize=12,
    linewidth=3,
    ax=ax1
)
ax1.set_title(r"Fraction of Bytes Transferred" + "\n" + r"(Pre-Agg. $\div$ no Pre-Agg.)", pad=20)
ax1.set_xlabel("Batch Size", labelpad=15)
ax1.set_ylabel("Fraction of Bytes Transferred", labelpad=15)
ax1.set_xticks(batch_sizes)
ax1.set_ylim(0.0, 1.5)
ax1.axhline(1.0, color="red", linestyle="--", linewidth=2, label="No difference", alpha=0.7)
ax1.legend(title="Dataset", frameon=True)

sns.lineplot(
    data=df, 
    x="Batch Size", 
    y="Relative Mean Latency", 
    hue="Dataset", 
    marker="s",
    markersize=12,
    linewidth=3,
    ax=ax2
)
ax2.set_title(r"Relative Mean Latency" + "\n" + r"(Pre-Agg. $\div$ no Pre-Agg.)", pad=20)
ax2.set_xlabel("Batch Size", labelpad=15)
ax2.set_ylabel("Relative Mean Latency", labelpad=15)
ax2.set_xticks(batch_sizes)
ax2.set_ylim(0.0, 1.5)
ax2.axhline(1.0, color="red", linestyle="--", linewidth=2, label="No difference", alpha=0.7)
ax2.legend(title="Dataset", frameon=True)

# Adjust layout
plt.tight_layout(pad=3.0)

# Save the plot
output_path = str(dir / f"preagg_{"_".join(map(str, FANOUT))}_bytes_and_latency_vs_batch_size.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_path}")
