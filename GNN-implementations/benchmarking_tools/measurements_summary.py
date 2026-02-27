from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd

PAIR_EVENTS: List[Tuple[str, str, str]] = [
    ("epoch_start", "epoch_end", "epoch_time_s"),
    ("start_batch_fetch", "end_batch_fetch", "sampling_phase_time_s"),
    ("start_batch_processing", "end_batch_processing", "training_phase_time_s"),
    ("start_validation_accuracy", "end_validation_accuracy", "validation_time_s"),
    ("start_saving_weights", "end_saving_weights", "saving_weights_time_s"),
    ("remote_feature_fetch", "remote_feature_recieved", "remote_feature_latency_s"),
]


@dataclass
class Stats:
    count: int = 0
    total_s: float = 0.0
    mean_s: Optional[float] = None
    p50_s: Optional[float] = None
    p90_s: Optional[float] = None
    p99_s: Optional[float] = None
    min_s: Optional[float] = None
    max_s: Optional[float] = None

    @staticmethod
    def from_series(s: pd.Series) -> "Stats":
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return Stats()
        return Stats(
            count=int(len(s)),
            total_s=float(s.sum()),
            mean_s=float(s.mean()),
            p50_s=float(s.quantile(0.50)),
            p90_s=float(s.quantile(0.90)),
            p99_s=float(s.quantile(0.99)),
            min_s=float(s.min()),
            max_s=float(s.max()),
        )


def read_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Event", "Time", "Value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)} (expected {sorted(expected)})")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def pair_durations(df: pd.DataFrame, start_event: str, end_event: str) -> pd.Series:
    starts = df.loc[df["Event"] == start_event, "Time"].to_list()
    ends = df.loc[df["Event"] == end_event, "Time"].to_list()
    n = min(len(starts), len(ends))
    if n == 0:
        return pd.Series(dtype="float64")
    starts = starts[:n]
    ends = ends[:n]
    d = [max(0.0, e - s) for s, e in zip(starts, ends)]
    return pd.Series(d, dtype="float64")


def get_validation_accuracies(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Event"] == "validation_accuracy"][["Time", "Value"]]


def _first_time(df: pd.DataFrame, event: str) -> Optional[float]:
    m = df.loc[df["Event"] == event, "Time"]
    return float(m.iloc[0]) if len(m) else None


def _last_time(df: pd.DataFrame, event: str) -> Optional[float]:
    m = df.loc[df["Event"] == event, "Time"]
    return float(m.iloc[-1]) if len(m) else None


def _last_value(df: pd.DataFrame, event: str) -> Optional[float]:
    m = df.loc[df["Event"] == event, "Value"]
    return float(m.iloc[-1]) if len(m) else None


def _best_value(df: pd.DataFrame, event: str, mode: str = "max") -> Optional[float]:
    m = df.loc[df["Event"] == event, "Value"]
    if len(m) == 0:
        return None
    m = pd.to_numeric(m, errors="coerce").dropna()
    if len(m) == 0:
        return None
    return float(m.max() if mode == "max" else m.min())


def _time_at_best(df: pd.DataFrame, event: str, mode: str = "max") -> Optional[float]:
    sub = df[df["Event"] == event].copy()
    if sub.empty:
        return None
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    sub = sub.dropna(subset=["Value"])
    if sub.empty:
        return None
    idx = sub["Value"].idxmax() if mode == "max" else sub["Value"].idxmin()
    return float(df.loc[idx, "Time"])


def build_summary(csv_path: Path, df: pd.DataFrame) -> Dict[str, Any]:
    program_start = _first_time(df, "program_start")
    program_end = _last_time(df, "program_end")
    if program_start is None:
        program_start = float(df["Time"].min())
    if program_end is None:
        program_end = float(df["Time"].max())
    runtime_s = float(program_end - program_start)

    first_epoch_start = _first_time(df, "epoch_start")
    last_epoch_end = _last_time(df, "epoch_end")
    training_start = _first_time(df, "start_batch_processing") or first_epoch_start
    if training_start is not None and last_epoch_end is not None and last_epoch_end > training_start:
        training_time_s = float(last_epoch_end - training_start)
    else:
        training_time_s = runtime_s

    epochs_completed = int((df["Event"] == "epoch_end").sum()) or int((df["Event"] == "epoch_start").sum())
    batches_seen = int((df["Event"] == "end_batch_fetch").sum()) or int((df["Event"] == "start_batch_fetch").sum())
    nbr_training_datapoints = _last_value(df, "nbr_training_datapoints")

    throughput_samples_per_s = None
    if nbr_training_datapoints is not None and training_time_s > 0 and epochs_completed > 0:
        throughput_samples_per_s = float((nbr_training_datapoints * epochs_completed) / training_time_s)

    converged_time = _first_time(df, "training_converged")
    converged_epoch = _last_value(df, "training_converged")
    convergence_time_s = float(converged_time - program_start) if (converged_time is not None and program_start is not None) else None

    final_val_acc = _last_value(df, "validation_accuracy")
    first_val_acc_time = _first_time(df, "validation_accuracy")
    best_val_acc = _best_value(df, "validation_accuracy", mode="max")
    time_at_best_acc = _time_at_best(df, "validation_accuracy", mode="max")
    time_to_best_acc_s = float(time_at_best_acc - program_start) if (time_at_best_acc is not None and program_start is not None) else None
    time_to_first_val_acc_s = float(first_val_acc_time - program_start) if (first_val_acc_time is not None and program_start is not None) else None

    final_val_loss = _last_value(df, "validation_loss")
    best_val_loss = _best_value(df, "validation_loss", mode="min")
    time_at_best_loss = _time_at_best(df, "validation_loss", mode="min")
    time_to_best_loss_s = float(time_at_best_loss - program_start) if (time_at_best_loss is not None and program_start is not None) else None

    final_test_acc = _last_value(df, "test_accuracy")
    best_test_acc = _best_value(df, "test_accuracy", mode="max")
    time_at_best_test_acc = _time_at_best(df, "test_accuracy", mode="max")
    time_to_best_test_acc_s = float(time_at_best_test_acc - program_start) if (time_at_best_test_acc is not None and program_start is not None) else None

    cache_hits = int((df["Event"] == "cache_hit").sum())
    cache_misses = int((df["Event"] == "cache_miss").sum())
    cache_total = cache_hits + cache_misses
    cache_hit_rate = float(cache_hits / cache_total) if cache_total > 0 else None

    paired_stats: Dict[str, Dict[str, Any]] = {}
    for s_ev, e_ev, name in PAIR_EVENTS:
        durations = pair_durations(df, s_ev, e_ev)
        paired_stats[name] = asdict(Stats.from_series(durations))

    batch_losses = df.loc[df["Event"] == "batch_train_loss", "Value"]
    batch_loss_mean = float(pd.to_numeric(batch_losses, errors="coerce").dropna().mean()) if len(batch_losses) else None

    remote_feature_total_s = paired_stats.get("remote_feature_latency_s", {}).get("total_s", 0.0) or 0.0

    cpu_values = df.loc[df["Event"] == "cpu_utilization_percentage", "Value"]
    cpu_values = pd.to_numeric(cpu_values, errors="coerce").dropna()
    avg_cpu_utilization = float(cpu_values.mean()) if len(cpu_values) else None

    return {
        "run": {
            "csv": str(csv_path),
            "runtime_s": runtime_s,
            "training_time_s": training_time_s,
            "epochs_completed": epochs_completed,
            "batches_seen": batches_seen,
            "nbr_training_datapoints": nbr_training_datapoints,
        },
        "metrics": {
            "throughput_samples_per_s": throughput_samples_per_s,
            "epoch_time_s": paired_stats["epoch_time_s"],
            "sampling_phase_time_s": paired_stats["sampling_phase_time_s"],
            "training_phase_time_s": paired_stats["training_phase_time_s"],
            "convergence_time_s": convergence_time_s,
            "converged_epoch": converged_epoch,
            "avg_cpu_utilization": avg_cpu_utilization,
            "final_validation_accuracy": final_val_acc,
            "best_validation_accuracy": best_val_acc,
            "time_to_first_validation_accuracy_s": time_to_first_val_acc_s,
            "time_to_best_accuracy_s": time_to_best_acc_s,
            "final_validation_loss": final_val_loss,
            "best_validation_loss": best_val_loss,
            "time_to_best_loss_s": time_to_best_loss_s,
            "final_test_accuracy": final_test_acc,
            "best_test_accuracy": best_test_acc,
            "time_to_best_test_accuracy_s": time_to_best_test_acc_s,
            "remote_feature_latency_s": paired_stats["remote_feature_latency_s"],
            "remote_feature_total_s": remote_feature_total_s,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "validation_time_s": paired_stats["validation_time_s"],
            "saving_weights_time_s": paired_stats["saving_weights_time_s"],
            "batch_train_loss_mean": batch_loss_mean,
        },
        "notes": [
            "GPU/CPU utilization and explicit DB write/query timings are not in the Measurer CSV (unless you log them); this parser will report None for those.",
            "throughput_samples_per_s assumes nbr_training_datapoints is per-epoch; if it's total across training, remove the * epochs_completed multiplier.",
            "remote_feature_total_s is a proxy for time spent waiting on remote feature reads (DB reads).",
        ],
    }
