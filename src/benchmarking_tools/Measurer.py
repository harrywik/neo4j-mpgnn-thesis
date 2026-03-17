import csv
import time
import json
from pathlib import Path
from typing import Any, Optional

from .config_writer import ConfigWriter
from .measurements_summary import read_measurements, build_summary, get_validation_accuracies
from .measurements_plots import plot_phase_summary, plot_validation_convergence, plot_validation_convergence_time, plot_cpu_utilization, plot_subphase_latency
from .QueryProfileAccumulator import QueryProfileAccumulator

class Measurer:
    def __init__(self, config: dict, profile_accumulator: Optional[QueryProfileAccumulator] = None):
        # Handles all results path logic internally
        results_path = Path(__file__).parent.parent.parent / "experiment_results" / "results"
        results_path.mkdir(parents=True, exist_ok=True)

        # Find next available run_N
        existing = []
        for p in results_path.iterdir():
            if p.is_dir() and p.name.startswith("run_"):
                try:
                    existing.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    pass
        next_id = (max(existing) + 1) if existing else 0
        run_results_path = results_path / f"run_{next_id}"
        run_results_path.mkdir(parents=True, exist_ok=False)

        # Save config
        self.config_writer = ConfigWriter(run_results_path, config)
        self.profile_accumulator = profile_accumulator

        self.measurements_path = run_results_path / "measurements.csv"
        self.run_results_path = run_results_path
        self._csvfile = open(self.measurements_path, "a", newline="\n")
        self._writer = csv.writer(self._csvfile)

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
        plot_validation_convergence_time(csv_path, df)
        plot_cpu_utilization(csv_path, df)
        if self.profile_accumulator is not None and self.profile_accumulator.has_data():
            # Inject derived DB-exec / network-transfer metrics into the main
            # summary so they appear in measurements.json as well.
            profile_summary = self.profile_accumulator.get_summary()
            for source, src_data in profile_summary.items():
                glb = src_data.get("global", {})
                for key in ("avg_db_exec_time_ms", "avg_network_transfer_ms",
                            "avg_driver_overhead_ms", "avg_result_consumed_after_ms",
                            "avg_client_wall_time_ms"):
                    value = glb.get(key)
                    if value is not None:
                        summary["metrics"][f"{source}_{key}"] = value

            # Expose sampler totals under their natural measurement names so
            # they appear alongside other per-call averages in measurements.json.
            sampler_glb = profile_summary.get("sampler", {}).get("global", {})
            if sampler_glb.get("avg_client_wall_time_ms") is not None:
                summary["metrics"]["topo_fetch_ms"] = sampler_glb["avg_client_wall_time_ms"]
            if sampler_glb.get("avg_driver_overhead_ms") is not None:
                summary["metrics"]["network_baseline_ms"] = sampler_glb["avg_driver_overhead_ms"]

        plot_subphase_latency(csv_path, summary)

        json_path = csv_path.with_suffix(".json")
        try:
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write JSON summary to {json_path}: {e}")

        if self.profile_accumulator is not None and self.profile_accumulator.has_data():
            profile_path = csv_path.with_name("query_profile.json")
            try:
                self.profile_accumulator.save(profile_path, subphase_metrics=summary.get("metrics"))
            except Exception as e:
                print(f"Warning: Failed to write query profile to {profile_path}: {e}")

        return summary
    
    def log_event(self, event_name: str, value: int | float = 1):
        self._writer.writerow([event_name, time.monotonic(), value])

    def close(self) -> None:
        """Flush and close measurement outputs explicitly."""
        if not hasattr(self, "measurements_path"):
            return
        try:
            self._csvfile.flush()
            self._csvfile.close()
        except Exception:
            pass

    def __del__(self):
        if not hasattr(self, "measurements_path"):
            return
        try:
            self.log_event("program_end", 1)
            self._csvfile.flush()
            self._csvfile.close()
        except Exception:
            pass


