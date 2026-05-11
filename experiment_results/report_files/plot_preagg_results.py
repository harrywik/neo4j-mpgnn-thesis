import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

from pathlib import Path

FANOUT = [10, 5]

dir = Path(__file__).resolve().parent / "rq_preagg"
batch_sizes = [32, 64, 128, 256, 512]

def build_df(dir: Path, batch_sizes: list[int], fanout: list[int]) -> pd.DataFrame:
    techniques = ["no_preagg", "preagg"]
    datasets = ["products", "citeseer"]
    dataset_displayname = {
        "products": "obgn-products",
        "citeseer": "CiteSeer"
    }
    data = []

    def parse_subdir(dir, dataset, batch_size, techniques, fanout) -> dict:
        fandir = fanout.join("_")
        baseline = json.parse(str(dir / techniques[0] / dataset / fandir / batch_size / "measurements.json"))
        improvement = json.parse(str(dir / techniques[1] / dataset / fandir / batch_size / "measurements.json"))

        a = baseline["avg_feat_bytes"]
        b = improvement["avg_feat_bytes"]
        c = baseline["remote_feature_latency_s"]["mean_s"]
        d = improvement["remote_feature_latency_s"]["mean_s"]

        return {
            "Batch Size": batch_size,
            "Bytes Fraction": (a - b) / a,
            "Relative Mean Latency": c / d,
            "Dataset": dataset_displayname[dataset]
        }



    for ds in datasets:
        for batch_size in batch_sizes:
            data.append(parse_subdir(dir, ds, batch_size, techniques, fanout))

    return pd.DataFrame(data)

df = pd.DataFrame(build_df(dir, batch_sizes, fanout=FANOUT))

sns.set_theme(style="whitegrid")
# Create the side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

sns.lineplot(
    data=df, 
    x="Batch Size", 
    y="Bytes Fraction", 
    hue="Dataset", 
    marker="o",
    linewidth=2.5,
    ax=ax1
)
ax1.set_title("Fraction of Bytes Transferred vs. Batch Size", fontsize=14, pad=15)
ax1.set_xlabel("Batch Size", fontsize=12)
ax1.set_ylabel("Fraction of Bytes Transferred", fontsize=12)
ax1.set_xticks(batch_sizes)
ax1.set_ylim(0, 100)
ax1.legend(title="Dataset")

sns.lineplot(
    data=df, 
    x="Batch Size", 
    y="Relative Mean Latency", 
    hue="Dataset", 
    marker="s",
    linewidth=2.5,
    ax=ax2
)
ax2.set_title("Relative Mean Latency vs. Batch Size", fontsize=14, pad=15)
ax2.set_xlabel("Batch Size", fontsize=12)
ax2.set_ylabel("Relative Mean Latency", fontsize=12)
ax2.set_xticks(batch_sizes)
ax2.legend(title="Dataset")

# Adjust layout
plt.tight_layout()

# Save the plot
output_path = str(dir.parent / "preagg_fanout_and_latency_vs_batch_size.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_path}")
