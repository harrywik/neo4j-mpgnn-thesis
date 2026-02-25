import csv
import time
import json
from pathlib import Path
from typing import Any

from .config_writer import ConfigWriter
from .measurements_summary import read_measurements, build_summary, get_validation_accuracies
from .measurements_plots import plot_phase_summary, plot_validation_convergence

class Measurer:
    def __init__(self, config: dict):
        # Handles all results path logic internally
        results_path = Path(__file__).parent.parent.parent / "experiments" / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        num_folders = sum(1 for p in results_path.iterdir() if p.is_dir())
        results_name = f"run_{num_folders}"
        run_results_path = results_path / results_name
        run_results_path.mkdir(parents=True, exist_ok=False)

        # Save config
        self.config_writer = ConfigWriter(run_results_path, config)

        self.measurements_path = run_results_path / "measurements.csv"
        self.run_results_path = run_results_path
        rows = [
            ["Event", "Time", "Value"],
            ["program_start", time.monotonic(), 1],
        ]
        # Write to a CSV file
        with open(self.measurements_path, "w", newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

    def write_to_configresult(self, key: str, value: Any) -> None:
        """Add or update a key in the stored config.json for this run."""
        self.config_writer.update(key, value)

    def summarize(self):
        """Summarize the measurements CSV and write a JSON summary in the same folder."""
        csv_path = self.measurements_path
        df = read_measurements(csv_path)
        summary = build_summary(csv_path, df)

        val_acc_path = csv_path.with_name("validation_accuracies.csv")
        val_accs = get_validation_accuracies(df)
        val_accs.to_csv(val_acc_path, index=False)

        plot_phase_summary(csv_path, df)
        plot_validation_convergence(csv_path, df)

        json_path = csv_path.with_suffix(".json")
        try:
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write JSON summary to {json_path}: {e}")
        return summary
    
    def log_event(self, event_name: str, value: int | float = 1):
        with open(self.measurements_path, "a", newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([event_name, time.monotonic(), value])
    
    def __del__(self):
        self.log_event("program_end", 1)
        try:
            self.summarize()
        except Exception as e:
            print(f"Warning: Failed to summarize measurements: {e}")

