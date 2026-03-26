"""Standalone summarisation script.

Finds the latest run_N folder under experiment_results/results/ and generates
all summary files (measurements.json, plots, validation_accuracies.csv) if they
do not already exist.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "experiment_results" / "results"

from benchmarking_tools.measurements_summary import read_measurements, build_summary, get_validation_accuracies
from benchmarking_tools.measurements_plots import (
    plot_phase_summary,
    plot_validation_convergence,
    plot_validation_convergence_time,
    plot_cpu_utilization,
    plot_subphase_latency,
    plot_all_operator_profiles,
    plot_driver_time_breakdown,
)


def find_latest_run(results_dir: Path) -> Path:
    runs = []
    for p in results_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            try:
                runs.append((int(p.name.split("_", 1)[1]), p))
            except ValueError:
                pass
    if not runs:
        raise FileNotFoundError(f"No run_N folders found in {results_dir}")
    return max(runs, key=lambda x: x[0])[1]


def summarise(run_dir: Path) -> None:
    csv_path = run_dir / "measurements.csv"
    json_path = run_dir / "measurements.json"
    profile_path = run_dir / "query_profile.json"

    if not csv_path.exists():
        print(f"No measurements.csv found in {run_dir}, skipping.")
        return

    other_dir   = run_dir / "other_plots"
    profile_dir = run_dir / "query_profile_plots"
    other_dir.mkdir(exist_ok=True)
    profile_dir.mkdir(exist_ok=True)

    # Operator-level plots from query_profile.json (has its own skip-if-exists logic).
    plot_all_operator_profiles(profile_path, output_dir=profile_dir)

    summary_files = [
        json_path,
        other_dir / "phase_summary.png",
        other_dir / "subphase_latency.png",
        run_dir   / "validation_accuracies.csv",
    ]
    if all(f.exists() for f in summary_files):
        print(f"Summary already exists for {run_dir.name}, nothing to do.")
        return

    print(f"Generating summary for {run_dir.name}...")

    df = read_measurements(csv_path)
    summary = build_summary(csv_path, df)

    val_accs = get_validation_accuracies(df)
    val_accs.to_csv(run_dir / "validation_accuracies.csv", index=False)

    plot_phase_summary(csv_path, df, output_dir=other_dir)
    plot_validation_convergence(csv_path, df, output_dir=other_dir)
    plot_validation_convergence_time(csv_path, df, output_dir=other_dir)
    plot_cpu_utilization(csv_path, df, output_dir=other_dir)
    plot_subphase_latency(csv_path, summary, output_dir=other_dir)

    prof_path = run_dir / "train_profile.prof"
    if prof_path.exists():
        n_batches = summary.get("run", {}).get("batches_seen") or 1
        plot_driver_time_breakdown(prof_path, n_batches, output_dir=other_dir)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Summary written to {run_dir.name}/")


if __name__ == "__main__":
    run_dir = find_latest_run(RESULTS_DIR)
    summarise(run_dir)
