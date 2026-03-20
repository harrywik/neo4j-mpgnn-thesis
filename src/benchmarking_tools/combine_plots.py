"""Combine matching plots from multiple experiment run folders side by side.

Usage (from repo root via make):
    make combine FILES="results/run_6 results/run_7"
    make combine FILES="results/run_6 arxiv/baseline_neo4j"

Or directly:
    python src/benchmarking_tools/combine_plots.py results/run_6 results/run_7

Each argument is a path relative to ``experiment_results/``.  For every PNG
filename that exists in *all* specified folders the script produces one combined
figure with the images placed side by side, one column per folder.  Output is
written to ``experiment_results/combined/combine_N/`` (auto-incremented),
preserving any subdirectory structure (e.g. ``query_profile_plots/``).

A ``config.json`` with a ``"dataset"`` key must exist in every folder; the
script aborts if datasets differ across folders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_RESULTS = REPO_ROOT / "experiment_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_combined_dir() -> Path:
    combined_base = EXPERIMENT_RESULTS / "combined"
    combined_base.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in combined_base.iterdir():
        if p.is_dir() and p.name.startswith("combine_"):
            try:
                existing.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                pass
    next_id = (max(existing) + 1) if existing else 0
    out = combined_base / f"combine_{next_id}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _read_dataset(folder: Path) -> str:
    config_path = folder / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in {folder}. "
            "Each folder must contain a config.json with a 'dataset' key."
        )
    with open(config_path) as f:
        cfg = json.load(f)
    dataset = cfg.get("dataset")
    if dataset is None:
        raise KeyError(
            f"'dataset' key not found in {config_path}. "
            "Cannot verify that folders are from the same dataset."
        )
    return str(dataset)


def _collect_pngs(folder: Path) -> set[str]:
    """Return relative PNG paths (as strings) found under *folder*."""
    return {
        str(p.relative_to(folder))
        for p in folder.rglob("*.png")
    }


def _combine_one(
    rel_path: str,
    folders: list[Path],
    labels: list[str],
    out_dir: Path,
) -> None:
    """Load the PNG from each folder and write a side-by-side figure."""
    n = len(folders)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, folder, label in zip(axes, folders, labels):
        img_path = folder / rel_path
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_title(label, fontsize=9, pad=4)
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: list[str]) -> None:
    if len(args) < 2:
        print(
            "Usage: python combine_plots.py <rel_path1> <rel_path2> [<rel_path3> ...]\n"
            "Paths are relative to experiment_results/.\n"
            "Example: python combine_plots.py results/run_6 results/run_7"
        )
        sys.exit(1)

    # Resolve folders
    folders: list[Path] = []
    labels: list[str] = args
    for arg in args:
        folder = EXPERIMENT_RESULTS / arg
        if not folder.is_dir():
            print(f"ERROR: folder not found: {folder}")
            sys.exit(1)
        folders.append(folder)

    # Read datasets (informational only — different datasets are allowed)
    datasets: list[str] = []
    for folder in folders:
        try:
            ds = _read_dataset(folder)
        except (FileNotFoundError, KeyError):
            ds = "unknown"
        datasets.append(ds)

    print(f"Combining {len(folders)} folders:")
    for label, folder, ds in zip(labels, folders, datasets):
        print(f"  [{label}]  {folder}  (dataset: {ds})")

    # Find common PNGs
    png_sets = [_collect_pngs(f) for f in folders]
    common = sorted(set.intersection(*png_sets))
    all_pngs = sorted(set.union(*png_sets))
    skipped = sorted(set(all_pngs) - set(common))

    if not common:
        print("\nNo PNG files with matching names found across all folders. Nothing to combine.")
        sys.exit(0)

    # Create output directory
    out_dir = _next_combined_dir()
    print(f"\nOutput: {out_dir}")
    print(f"Combining {len(common)} plot(s)...")

    for rel_path in common:
        _combine_one(rel_path, folders, labels, out_dir)
        print(f"  {rel_path}")

    if skipped:
        print(f"\nSkipped {len(skipped)} plot(s) not present in all folders:")
        for rel_path in skipped:
            present_in = [lbl for lbl, ps in zip(labels, png_sets) if rel_path in ps]
            print(f"  {rel_path}  (only in: {', '.join(present_in)})")

    print(f"\nDone. {len(common)} combined plot(s) written to {out_dir.name}/")


if __name__ == "__main__":
    main(sys.argv[1:])
