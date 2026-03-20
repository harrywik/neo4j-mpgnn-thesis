from __future__ import annotations

import io
import json
import pstats
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

from .measurements_summary import pair_durations, get_validation_accuracies


# ---------------------------------------------------------------------------
# Operator-profile helpers
# ---------------------------------------------------------------------------

def _strip_db(op_type: str) -> str:
    """Remove '@database' suffix from an operator type string."""
    return op_type.split("@")[0]


def _pipeline_num(pipeline: str) -> int:
    m = re.search(r"(\d+)", pipeline)
    return int(m.group(1)) if m else -1


def _op_label(op: dict) -> str:
    return f"{_strip_db(op['operator_type'])} [{op['id']}]"


def _plot_operator_rows_comparison(output_dir: Path, source: str, operators: list) -> None:
    """Grouped horizontal bars: actual rows vs planner-estimated rows per operator."""
    rows = [
        (op, op["avg_rows"], op["avg_estimated_rows"])
        for op in operators
        if op["avg_rows"] > 0 or op["avg_estimated_rows"] > 0
    ]
    if not rows:
        return
    rows.sort(key=lambda x: x[1], reverse=True)

    labels = [_op_label(op) for op, _, _ in rows]
    actual = [r[1] for r in rows]
    estimated = [r[2] for r in rows]
    height = 0.35
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    ax.barh(y_pos - height / 2, actual,    height, label="Actual rows",       color="#4C78A8")
    ax.barh(y_pos + height / 2, estimated, height, label="Planner estimate", color="#F58518", alpha=0.75)

    max_val = max(actual + estimated, default=1)
    if max_val > 0:
        ax.set_xscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_title(f"Actual vs estimated rows — {source}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"op_rows_{source}.png", dpi=150)
    plt.close(fig)


def _plot_operator_time_waterfall(output_dir: Path, source: str, operators: list) -> None:
    """Horizontal bars of per-operator time, coloured by pipeline stage."""
    rows = [(op, op["avg_time_ms"]) for op in operators if op["avg_time_ms"] > 0.001]
    if not rows:
        return
    rows.sort(key=lambda x: x[1], reverse=True)

    pipeline_ids = sorted({_pipeline_num(op["pipeline"]) for op, _ in rows})
    cmap = plt.cm.tab20
    pipeline_color = {
        p: cmap(i / max(len(pipeline_ids) - 1, 1))
        for i, p in enumerate(pipeline_ids)
    }

    labels = [_op_label(op) for op, _ in rows]
    times  = [t for _, t in rows]
    colors = [pipeline_color[_pipeline_num(op["pipeline"])] for op, _ in rows]

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    bars = ax.barh(y_pos, times, color=colors)

    max_t = max(times)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_width() + max_t * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} ms",
            va="center", ha="left", fontsize=7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean time per batch (ms)")
    ax.set_title(f"Operator time breakdown — {source}")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(color=pipeline_color[p], label=f"Pipeline {p}")
        for p in pipeline_ids
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / f"op_time_waterfall_{source}.png", dpi=150)
    plt.close(fig)


def _plot_operator_db_hits_scatter(output_dir: Path, source: str, operators: list) -> None:
    """Scatter: DB hits (y) vs rows produced (x), one point per operator."""
    rows = [
        (op, op["avg_rows"], op["avg_db_hits"])
        for op in operators
        if op["avg_rows"] > 0 or op["avg_db_hits"] > 0
    ]
    if not rows:
        return

    # Group operators by their base type (text before first '(' or ' ')
    def _family(op: dict) -> str:
        base = _strip_db(op["operator_type"])
        return re.split(r"[\( ]", base)[0]

    families = sorted({_family(op) for op, _, _ in rows})
    cmap = plt.cm.tab10
    fam_color = {f: cmap(i / max(len(families) - 1, 1)) for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for fam in families:
        pts = [(op, r, d) for op, r, d in rows if _family(op) == fam]
        xs = [r for _, r, _ in pts]
        ys = [d for _, _, d in pts]
        ax.scatter(xs, ys, color=fam_color[fam], label=fam, s=60, alpha=0.85, zorder=3)
        for op, r, d in pts:
            ax.annotate(
                f"[{op['id']}]", (r, d),
                textcoords="offset points", xytext=(4, 3),
                fontsize=6, color=fam_color[fam],
            )

    all_vals = [v for _, r, d in rows for v in (r, d) if v > 0]
    if all_vals and max(all_vals) / max(min(all_vals), 1) > 100:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Avg rows produced")
    ax.set_ylabel("Avg DB hits")
    ax.set_title(f"DB hits vs rows produced — {source}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"op_db_hits_{source}.png", dpi=150)
    plt.close(fig)


def _plot_operator_memory_profile(output_dir: Path, source: str, operators: list) -> None:
    """Horizontal bars of per-operator memory usage."""
    rows = [(op, op["avg_memory_bytes"]) for op in operators if op["avg_memory_bytes"] > 0]
    if not rows:
        return
    rows.sort(key=lambda x: x[1], reverse=True)

    labels = [_op_label(op) for op, _ in rows]
    values_kb = [v / 1024 for _, v in rows]

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    bars = ax.barh(y_pos, values_kb, color="#72B7B2")

    max_v = max(values_kb)
    for bar, val in zip(bars, values_kb):
        label_str = f"{val:.1f} KB" if val < 1024 else f"{val / 1024:.2f} MB"
        ax.text(
            bar.get_width() + max_v * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label_str,
            va="center", ha="left", fontsize=7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean memory per batch (KB)")
    ax.set_title(f"Memory profile per operator — {source}")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"op_memory_{source}.png", dpi=150)
    plt.close(fig)


def plot_all_operator_profiles(profile_path: Path, output_dir: Path | None = None) -> None:
    """Generate all 4 operator-level plots for every query source in *profile_path*.

    Saves plots to *output_dir* when provided, otherwise to the same directory
    as *profile_path*.  Skips a source if all 4 of its plots already exist.
    Safe to call when *profile_path* does not exist (no-op).
    """
    if not profile_path.exists():
        return

    with open(profile_path) as f:
        profile = json.load(f)

    out = output_dir or profile_path.parent

    for source, data in profile.items():
        if source == "subphase_metrics":
            continue
        operators = data.get("operators", [])
        if not operators:
            continue

        plot_names = [
            out / f"op_rows_{source}.png",
            out / f"op_time_waterfall_{source}.png",
            out / f"op_db_hits_{source}.png",
            out / f"op_memory_{source}.png",
        ]
        if all(p.exists() for p in plot_names):
            continue

        _plot_operator_rows_comparison(out, source, operators)
        _plot_operator_time_waterfall(out, source, operators)
        _plot_operator_db_hits_scatter(out, source, operators)
        _plot_operator_memory_profile(out, source, operators)
        print(f"  Operator plots generated for source '{source}'.")


def plot_phase_summary(csv_path: Path, df: pd.DataFrame, output_dir: Path | None = None) -> None:
    sampling_durations = pair_durations(df, "start_batch_fetch", "end_batch_fetch")
    training_durations = pair_durations(df, "start_batch_processing", "end_batch_processing")

    if len(sampling_durations) or len(training_durations):
        sampling_mean = float(sampling_durations.mean()) if len(sampling_durations) else 0.0
        training_mean = float(training_durations.mean()) if len(training_durations) else 0.0

        labels = ["sampling", "training"]
        means = [sampling_mean, training_mean]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, color=["#4C78A8", "#F58518"])
        ax.set_title("Phase durations (mean)")
        ax.set_ylabel("seconds")

        fig.tight_layout()
        out = (output_dir or csv_path.parent) / "phase_summary.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


def plot_validation_convergence(csv_path: Path, df: pd.DataFrame, output_dir: Path | None = None) -> None:
    val_accs = get_validation_accuracies(df).copy()
    if len(val_accs):
        val_accs = val_accs.reset_index(drop=True)
        epochs = np.arange(1, len(val_accs) + 1, dtype=float)
        acc_values = pd.to_numeric(val_accs["Value"], errors="coerce")
        mask = acc_values.notna()
        if not mask.any():
            return

        epochs = epochs[mask.to_numpy()]
        acc_values = acc_values[mask].to_numpy(dtype=float)

        color = "#E45756"
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(epochs, acc_values, color=color, linewidth=1.8)
        ax.scatter(epochs, acc_values, color=color, s=18, zorder=3)

        # Mark best epoch with a dashed vertical line
        best_idx = int(np.argmax(acc_values))
        best_epoch = float(epochs[best_idx])
        ax.axvline(x=best_epoch, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            best_epoch, 0.02, f"{int(round(best_epoch))}",
            color=color, fontsize=7, ha="center", va="bottom",
            transform=ax.get_xaxis_transform(),
        )

        ax.set_title("Validation accuracy vs epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        out = (output_dir or csv_path.parent) / "validation_convergence.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)


def plot_validation_convergence_time(csv_path: Path, df: pd.DataFrame, output_dir: Path | None = None) -> None:
    val_accs = get_validation_accuracies(df).copy()
    if val_accs.empty:
        return

    val_accs = val_accs.reset_index(drop=True)
    times = pd.to_numeric(val_accs["Time"], errors="coerce")
    acc_values = pd.to_numeric(val_accs["Value"], errors="coerce")
    mask = times.notna() & acc_values.notna()
    if not mask.any():
        return

    times = times[mask]
    acc_values = acc_values[mask]

    epoch_start = df.loc[df["Event"] == "epoch_start", "Time"]
    if len(epoch_start):
        t0 = float(epoch_start.iloc[0])
    else:
        t0 = float(times.min())

    times = (times - t0).to_numpy(dtype=float)
    acc_values = acc_values.to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(acc_values) & (times >= 0)
    times = times[valid]
    acc_values = acc_values[valid]
    if len(times) == 0:
        return

    color = "#E45756"
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, acc_values, color=color, linewidth=1.8)
    ax.scatter(times, acc_values, color=color, s=18, zorder=3)

    # Mark best epoch with a dashed vertical line
    best_idx = int(np.argmax(acc_values))
    ax.axvline(x=times[best_idx], color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        times[best_idx], 0.02, f"{times[best_idx]:.1f}s",
        color=color, fontsize=7, ha="center", va="bottom",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title("Validation accuracy vs wall time")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = (output_dir or csv_path.parent) / "validation_convergence_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_driver_time_breakdown(
    prof_path: Path,
    n_batches: int,
    output_dir: Path | None = None,
) -> None:
    """Horizontal bar chart: where the Neo4j Python driver spends time per batch.

    Reads the binary cProfile ``.prof`` file saved by ``Training.train()``,
    extracts ``tottime`` for key leaf functions (socket I/O, Bolt decode,
    packstream, NumPy reconstruction, Python record objects), divides by
    ``n_batches`` to get mean ms per batch, and renders a single-row waterfall
    styled bar chart comparable to :func:`plot_subphase_latency_waterfall`.

    All entries are **leaf** ``tottime`` values so there is no double-counting.
    The total shown is the sum of these leaves; remaining driver overhead (Bolt
    frame routing, call-chain glue) is not attributed to any leaf and is noted
    in the subtitle.

    Saved as ``driver_time_breakdown.png``.
    """
    if not prof_path.exists():
        return

    # Load pstats from the binary profile
    try:
        st = pstats.Stats(str(prof_path), stream=io.StringIO())
        st.strip_dirs()
    except Exception:
        return

    # stats.stats is a dict:
    #   (file, line, func) → (pcalls, ncalls, tottime, cumtime, callers)
    raw = st.stats  # type: ignore[attr-defined]

    # Lookup helpers: match by (filename_fragment, function_name)
    def _find_tottime(file_frag: str, func_name: str) -> float:
        for (fname, _line, fn), entry in raw.items():
            if file_frag in fname and fn == func_name:
                return entry[2]  # tottime
        return 0.0

    entries = [
        ("Socket receive\n(recv_into)",       "_socket",      "recv_into",             "#4C78A8"),
        ("Socket overhead\n(settimeout)",      "_socket",      "settimeout",            "#7BAFD4"),
        ("Packstream decode\n(unpack)",        "packstream",   "unpack",                "#AED4F0"),
        ("byte[] → ndarray\n(frombuffer)",     "fromnumeric",  "frombuffer",            "#9ECAE1"),
        ("Record obj. creation\n(__new__)",    "_data",        "__new__",               "#6BAED6"),
    ]

    # frombuffer is a built-in — try alternate key patterns
    _frombuffer_total = _find_tottime("fromnumeric", "frombuffer")
    if _frombuffer_total == 0.0:
        # built-in method entry stored differently
        for (fname, _line, fn), entry in raw.items():
            if "frombuffer" in fn:
                _frombuffer_total = max(_frombuffer_total, entry[2])

    totals_s: list[tuple[str, float, str]] = []
    for label, file_frag, func_name, color in entries:
        if func_name == "frombuffer":
            t = _frombuffer_total
        else:
            t = _find_tottime(file_frag, func_name)
        if t > 0:
            totals_s.append((label, t, color))

    if not totals_s:
        return

    # Convert to ms per batch
    ms_per_batch = [(lbl, t * 1000.0 / n_batches, col) for lbl, t, col in totals_s]
    total_ms = sum(v for _, v, _ in ms_per_batch)

    fig, ax = plt.subplots(figsize=(max(6, len(ms_per_batch) * 1.6), 2.8))
    left = 0.0
    legend_patches = []
    for label, width, color in ms_per_batch:
        if width <= 0:
            continue
        ax.barh(0, width, left=left, height=0.5, color=color,
                edgecolor="white", linewidth=0.5)
        frac = width / total_ms
        if frac > 0.05:
            short_label = label.split("\n")[0]
            ax.text(
                left + width / 2, 0,
                f"{width:.2f}",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )
        left += width
        legend_patches.append(Patch(facecolor=color, label=label.replace("\n", " "), edgecolor="white"))

    # Total label at right edge
    ax.text(left * 1.01, 0, f"{total_ms:.2f} ms", ha="left", va="center", fontsize=8)

    ax.set_yticks([0])
    ax.set_yticklabels(["Driver\nleaf cost"], fontsize=8)
    ax.set_xlabel("mean ms per batch (leaf tottime / n_batches)")
    ax.set_title(
        "Neo4j driver time breakdown (cProfile leaf functions)",
        fontsize=9,
    )
    ax.set_xlim(0, left * 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              framealpha=0.85, ncol=1)
    fig.tight_layout()
    out = (output_dir or prof_path.parent) / "driver_time_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_subphase_latency_waterfall(csv_path: Path, summary: dict, output_dir: Path | None = None) -> None:
    """Waterfall / Gantt-style timeline bar for per-batch sub-phase latencies.

    Each phase is rendered as a segment **starting where the previous one ended**,
    so the visual length of each segment is the *exclusive* cost of that phase and
    the right edge of the last segment is the true total latency.  This avoids the
    overlapping-bars ambiguity of :func:`plot_subphase_latency`.

    Topology row (left → right):
      DB: Cypher execution  |  result transfer  |  driver / Python overhead  |  Python ETL

    Features row (left → right):
      DB: property I/O  |  result transfer  |  driver / Python overhead  |  Python ETL

    Saved as ``subphase_latency_waterfall.png`` next to the CSV.
    """
    metrics = summary.get("metrics", {})

    def _get(*keys: str) -> float | None:
        for k in keys:
            v = metrics.get(k)
            if v is not None:
                return float(v)
        return None

    # ── Topology slices ──────────────────────────────────────────────────────
    topo_wall     = _get("topo_fetch_ms", "sampler_avg_client_wall_time_ms")
    topo_db       = _get("sampler_avg_db_exec_time_ms")
    topo_consumed = _get("sampler_avg_result_consumed_after_ms")
    topo_etl      = _get("topo_etl_ms")

    # Shared neutral colours so that "Result transfer", "Driver / Python recv"
    # and "Python ETL" look identical across both rows and only need one legend
    # entry each regardless of which row is rendered last.
    _C_TRANSFER = "#BDBDBD"   # medium grey  – result transfer
    _C_DRIVER   = "#D9D9D9"   # light grey   – driver / Python recv
    _C_ETL      = "#F0F0F0"   # very light grey – Python ETL

    topo_slices: list[tuple[str, float, str]] = []
    if topo_db is not None:
        topo_slices.append(("DB: Cypher execution", topo_db, "#4C78A8"))
        if topo_consumed is not None:
            transfer = max(0.0, topo_consumed - topo_db)
            topo_slices.append(("Result transfer", transfer, _C_TRANSFER))
            if topo_wall is not None:
                driver = max(0.0, topo_wall - topo_consumed)
                topo_slices.append(("Driver / Python recv", driver, _C_DRIVER))
        elif topo_wall is not None:
            driver = max(0.0, topo_wall - topo_db)
            topo_slices.append(("Driver / Python recv", driver, _C_DRIVER))
    elif topo_wall is not None:
        topo_slices.append(("Total fetch (wall)", topo_wall, "#2171B5"))
    if topo_etl is not None:
        topo_slices.append(("Python ETL", topo_etl, _C_ETL))

    # ── Feature slices ───────────────────────────────────────────────────────
    feat_wall     = _get("feat_x_avg_client_wall_time_ms")
    feat_db       = _get("feat_x_avg_db_exec_time_ms")
    feat_consumed = _get("feat_x_avg_result_consumed_after_ms")
    feat_driver   = _get("feat_x_avg_driver_overhead_ms")
    feat_etl      = _get("feat_x_etl_ms")

    feat_slices: list[tuple[str, float, str]] = []
    if feat_db is not None:
        feat_slices.append(("DB: property I/O", feat_db, "#F58518"))
        if feat_consumed is not None:
            transfer = max(0.0, feat_consumed - feat_db)
            feat_slices.append(("Result transfer", transfer, _C_TRANSFER))
            if feat_wall is not None:
                driver = max(0.0, feat_wall - feat_consumed)
                feat_slices.append(("Driver / Python recv", driver, _C_DRIVER))
        elif feat_wall is not None:
            driver = max(0.0, feat_wall - feat_db)
            feat_slices.append(("Driver / Python recv", driver, _C_DRIVER))
    elif feat_wall is not None:
        feat_slices.append(("Total fetch (wall)", feat_wall, "#D94F00"))
    if feat_etl is not None:
        feat_slices.append(("Python ETL", feat_etl, _C_ETL))

    rows = []
    if topo_slices:
        rows.append(("Topology", topo_slices))
    if feat_slices:
        rows.append(("Features", feat_slices))

    if not rows:
        return

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(2.5, n_rows * 1.4)))

    bar_height = 0.5
    yticks, ylabels = [], []

    # Collect all unique slice labels for the legend
    legend_handles: dict[str, tuple[str, str]] = {}  # label → (label, color)

    for row_i, (row_label, slices) in enumerate(rows):
        y = row_i
        left = 0.0
        for seg_label, width, color in slices:
            if width > 0:
                ax.barh(y, width, left=left, height=bar_height, color=color,
                        edgecolor="white", linewidth=0.5)
                # Label segment if wide enough to be readable (> 3% of total)
                total = sum(s[1] for s in slices)
                if width / total > 0.03:
                    ax.text(
                        left + width / 2, y,
                        f"{width:.1f}",
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold",
                    )
                left += width
                legend_handles[seg_label] = (seg_label, color)

        # Annotate total at the right edge
        ax.text(left + (ax.get_xlim()[1] * 0.005 if ax.get_xlim()[1] > 0 else 5),
                y, f"{left:.1f} ms",
                ha="left", va="center", fontsize=8)

        yticks.append(y)
        ylabels.append(row_label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch (cumulative timeline)")
    ax.set_title("Sub-phase latency breakdown (waterfall)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    legend_patches = [
        Patch(facecolor=color, label=label, edgecolor="white")
        for label, color in legend_handles.values()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              framealpha=0.85, ncol=1)

    # Re-apply xlim with a small right margin for the total label
    max_total = max(sum(s[1] for s in slices) for _, slices in rows)
    ax.set_xlim(0, max_total * 1.18)

    fig.tight_layout()
    out = (output_dir or csv_path.parent) / "subphase_latency_waterfall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_subphase_latency(csv_path: Path, summary: dict, output_dir: Path | None = None) -> None:
    """Bar chart of per-batch sub-phase latencies from a single run.

    Reads the scalar fields written by the micro-timers in
    ``BaseLineGS`` / ``NoCacheFeatureStore`` and produces a horizontal bar
    chart saved as ``subphase_latency.png`` next to the CSV.
    """
    metrics = summary.get("metrics", {})

    topo_segments = [
        ("topo_fetch_ms",               "Topology: total fetch (wall)",              "#2171B5"),
        ("sampler_avg_db_exec_time_ms", "Topology: DB Cypher execution",             "#4C78A8"),
        ("network_baseline_ms",         "Topology: transfer + driver overhead",      "#7BAFD4"),
        ("topo_etl_ms",                 "Topology: Python ETL",                      "#AED4F0"),
    ]
    # x and y share one query, so all timings cover both in a single round-trip.
    # The profiler attributes the combined query to "feat_x" internally.
    feat_segments = [
        ("feat_x_avg_client_wall_time_ms", "Features ([float], int): total fetch",                    "#D94F00"),
        ("feat_x_avg_db_exec_time_ms",     "Features ([float], int): DB property I/O",               "#F58518"),
        ("feat_y_avg_db_exec_time_ms",     "Features ([float], int): DB property I/O (y)",            "#54A24B"),
        ("feat_x_avg_driver_overhead_ms",  "Features ([float], int): transfer + driver overhead",     "#F7A850"),
        ("feat_x_etl_ms",                  "Features ([float], int): Python ETL",                     "#FAC980"),
    ]

    def _resolve(segs):
        rows = []
        for key, label, color in segs:
            v = metrics.get(key)
            if v is not None:
                rows.append((label, float(v), color))
        return sorted(rows, key=lambda r: r[1], reverse=True)

    topo_rows = _resolve(topo_segments)
    feat_rows = _resolve(feat_segments)
    all_rows = topo_rows + feat_rows

    if not all_rows:
        return

    labels = [r[0] for r in all_rows]
    values = [r[1] for r in all_rows]
    colors = [r[2] for r in all_rows]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} ms",
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch")
    ax.set_title("Sub-phase latency breakdown")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = (output_dir or csv_path.parent) / "subphase_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cpu_utilization(csv_path: Path, df: pd.DataFrame, output_dir: Path | None = None) -> None:
    cpu = df[df["Event"] == "cpu_utilization_percentage"][["Time", "Value"]].copy()
    ram = df[df["Event"] == "ram_usage_mb"][["Time", "Value"]].copy()
    if cpu.empty:
        return

    program_start = df.loc[df["Event"] == "program_start", "Time"]
    program_end = df.loc[df["Event"] == "program_end", "Time"]
    if len(program_start):
        t0 = float(program_start.iloc[0])
    else:
        t0 = float(df["Time"].min())

    if len(program_end):
        t1 = float(program_end.iloc[-1])
    else:
        t1 = float(df["Time"].max())

    if t1 <= t0:
        return

    cpu["rel_time"] = pd.to_numeric(cpu["Time"], errors="coerce") - t0
    cpu = cpu.dropna(subset=["rel_time", "Value"]).sort_values("rel_time")
    if cpu.empty:
        return

    x = cpu["rel_time"].to_numpy()
    y = pd.to_numeric(cpu["Value"], errors="coerce").to_numpy()
    if len(x) == 0:
        return

    # Interpolate to a uniform grid from 0 to duration
    duration = t1 - t0
    grid = np.linspace(0.0, duration, num=200)
    y_interp = np.interp(grid, x, y)

    fig, ax = plt.subplots(figsize=(6, 4))

    def _shade(event_start: str, event_end: str, color: str, alpha: float) -> None:
        starts = df.loc[df["Event"] == event_start, "Time"].to_list()
        ends = df.loc[df["Event"] == event_end, "Time"].to_list()
        n = min(len(starts), len(ends))
        for s, e in zip(starts[:n], ends[:n]):
            ax.axvspan(s - t0, e - t0, color=color, alpha=alpha, lw=0, zorder=0)

    # Sampling = light blue background
    _shade("start_batch_fetch", "end_batch_fetch", color="#DCEEFF", alpha=0.4)
    # Training = white background (subtle overlay to preserve contrast)
    _shade("start_batch_processing", "end_batch_processing", color="#FFFFFF", alpha=0.2)

    ax.plot(grid, y_interp, linestyle="-", zorder=2, label="CPU %")
    ax.set_title("CPU/RAM utilization over time")
    ax.set_xlabel("seconds")
    ax.set_ylabel("cpu utilization (%)")
    ax.set_xlim(0.0, duration)

    # RAM on secondary axis
    if not ram.empty:
        ram["rel_time"] = pd.to_numeric(ram["Time"], errors="coerce") - t0
        ram = ram.dropna(subset=["rel_time", "Value"]).sort_values("rel_time")
        if not ram.empty:
            rx = ram["rel_time"].to_numpy()
            ry = pd.to_numeric(ram["Value"], errors="coerce").to_numpy()
            if len(rx):
                rgrid = np.linspace(0.0, duration, num=200)
                r_interp = np.interp(rgrid, rx, ry)
                ax2 = ax.twinx()
                ax2.plot(rgrid, r_interp, linestyle="--", color="#2CA02C", label="RAM (MB)")
                ax2.set_ylabel("ram usage (MB)")

    # Background legend/comment
    ax.text(
        0.01,
        0.98,
        "Background: light blue = sampling, white = training",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        zorder=3,
    )

    fig.tight_layout()
    out = (output_dir or csv_path.parent) / "cpu_utilization.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# End-to-end latency: combined query_profile.json + train_profile.txt
# ---------------------------------------------------------------------------

_CPROFILE_LINE_RE = re.compile(
    r"^\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)$"
)

# Framework files whose "forward" functions we do NOT want to pick as the
# user's GNN model forward pass.
_FRAMEWORK_FORWARD_RE = re.compile(
    r"gcn_conv|message_passing|linear\.py|propagate|conv\.py|"
    r"basic\.py|base\.py|module\.py|experimental\.py|loop\.py",
    re.IGNORECASE,
)


def _parse_train_profile(txt_path: Path) -> dict:
    """Parse cProfile text output into a structured dict.

    Returns ``{"total_s": float, "functions": [{"ncalls", "tottime",
    "cumtime", "location"}]}`` or an empty dict if the file is absent.
    """
    if not txt_path.exists():
        return {}

    with open(txt_path) as f:
        content = f.read()

    total_match = re.search(r"in\s+([\d.]+)\s+seconds", content)
    total_s = float(total_match.group(1)) if total_match else None

    functions: list[dict] = []
    for line in content.splitlines():
        m = _CPROFILE_LINE_RE.match(line)
        if m:
            ncalls_str, tottime, _, cumtime, _, location = m.groups()
            functions.append({
                "ncalls":   int(ncalls_str.split("/")[0]),
                "tottime":  float(tottime),
                "cumtime":  float(cumtime),
                "location": location.strip(),
            })

    return {"total_s": total_s, "functions": functions}


def _find_profile_fn(functions: list[dict], *patterns: str) -> dict | None:
    """Return the first function whose location matches any regex pattern."""
    for fn in functions:
        loc = fn["location"]
        for p in patterns:
            if re.search(p, loc):
                return fn
    return None


def plot_end_to_end_latency(
    profile_json_path: Path,
    train_profile_path: Path,
    summary: dict,
    output_dir: Path | None = None,
) -> None:
    """Combined end-to-end per-batch latency waterfall.

    Three rows (rows are omitted when data is unavailable):

    * **Topology** – DB Cypher exec | result transfer | driver recv | Python ETL
    * **Features** – DB property I/O | result transfer | driver recv | Python ETL
    * **GNN compute** – model forward | backward | optimizer step

    Topology and Features come from ``query_profile.json``; GNN timings
    come from ``train_profile.txt`` (cProfile), normalised per training batch.

    Saved as ``end_to_end_latency.png``.
    """
    # ── Load query_profile.json ───────────────────────────────────────────
    qp: dict = {}
    if profile_json_path.exists():
        with open(profile_json_path) as f:
            qp = json.load(f)

    def _qp_global(source: str) -> dict:
        return qp.get(source, {}).get("global", {})

    metrics = summary.get("metrics", {})

    # Shared neutral colours for network/driver segments (same as waterfall).
    _C_TRANSFER = "#BDBDBD"
    _C_DRIVER   = "#D9D9D9"
    _C_ETL      = "#F0F0F0"

    # ── Topology slices ───────────────────────────────────────────────────
    sg = _qp_global("sampler")
    topo_db       = sg.get("avg_db_exec_time_ms")
    topo_transfer = sg.get("avg_network_transfer_ms")
    topo_driver   = sg.get("avg_driver_overhead_ms")
    topo_etl      = metrics.get("topo_etl_ms")

    topo_slices: list[tuple[str, float, str]] = []
    if topo_db is not None:
        topo_slices.append(("DB: Cypher execution", topo_db, "#4C78A8"))
    if topo_transfer is not None and topo_transfer > 0:
        topo_slices.append(("Result transfer", topo_transfer, _C_TRANSFER))
    if topo_driver is not None and topo_transfer is not None:
        pure_driver = max(0.0, topo_driver - topo_transfer)
        if pure_driver > 0:
            topo_slices.append(("Driver / Python recv", pure_driver, _C_DRIVER))
    if topo_etl is not None:
        topo_slices.append(("Python ETL", float(topo_etl), _C_ETL))

    # ── Feature slices ────────────────────────────────────────────────────
    fg = _qp_global("feat_x")
    feat_db       = fg.get("avg_db_exec_time_ms")
    feat_transfer = fg.get("avg_network_transfer_ms")
    feat_driver   = fg.get("avg_driver_overhead_ms")
    feat_etl      = metrics.get("feat_x_etl_ms")

    feat_slices: list[tuple[str, float, str]] = []
    if feat_db is not None:
        feat_slices.append(("DB: property I/O", feat_db, "#F58518"))
    if feat_transfer is not None and feat_transfer > 0:
        feat_slices.append(("Result transfer", feat_transfer, _C_TRANSFER))
    if feat_driver is not None and feat_transfer is not None:
        pure_driver = max(0.0, feat_driver - feat_transfer)
        if pure_driver > 0:
            feat_slices.append(("Driver / Python recv", pure_driver, _C_DRIVER))
    if feat_etl is not None:
        feat_slices.append(("Python ETL", float(feat_etl), _C_ETL))

    # ── GNN slices (from train_profile.txt) ──────────────────────────────
    gnn_slices: list[tuple[str, float, str]] = []
    profile_data = _parse_train_profile(train_profile_path)
    if profile_data:
        fns = profile_data["functions"]

        # Number of training batches = ncalls of _run_batch.
        run_batch_fn = _find_profile_fn(fns, r"Training\.py.*_run_batch")
        n_train = run_batch_fn["ncalls"] if run_batch_fn else None

        if n_train and n_train > 0:
            # GNN forward: highest-cumtime forward() not from a framework file.
            forward_candidates = [
                fn for fn in fns
                if re.search(r"\(forward\)$", fn["location"])
                and not _FRAMEWORK_FORWARD_RE.search(fn["location"])
            ]
            if forward_candidates:
                fwd_fn = max(forward_candidates, key=lambda f: f["cumtime"])
                ms_per_batch = fwd_fn["cumtime"] * 1000 / fwd_fn["ncalls"]
                if ms_per_batch > 0:
                    gnn_slices.append(("GNN forward", ms_per_batch, "#54A24B"))

            # Backward pass.
            bwd_fn = _find_profile_fn(
                fns,
                r"_tensor\.py.*\(backward\)",
                r"graph\.py.*_engine_run_backward",
                r"__init__.*\(backward\)",
            )
            if bwd_fn:
                ms_per_batch = bwd_fn["cumtime"] * 1000 / max(bwd_fn["ncalls"], 1)
                if ms_per_batch > 0:
                    gnn_slices.append(("GNN backward", ms_per_batch, "#88D27A"))

            # Optimizer step.
            opt_fn = _find_profile_fn(
                fns,
                r"optimizer\.py.*\(wrapper\)",
                r"adam\.py.*\(step\)",
                r"adam\.py.*\(adam\)",
            )
            if opt_fn:
                ms_per_batch = opt_fn["cumtime"] * 1000 / max(opt_fn["ncalls"], 1)
                if ms_per_batch > 0:
                    gnn_slices.append(("Optimizer step", ms_per_batch, "#B8EFB0"))

    # ── Assemble rows ─────────────────────────────────────────────────────
    rows: list[tuple[str, list]] = []
    if topo_slices:
        rows.append(("Topology", topo_slices))
    if feat_slices:
        rows.append(("Features", feat_slices))
    if gnn_slices:
        rows.append(("GNN compute", gnn_slices))

    if not rows:
        return

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(2.5, n_rows * 1.5)))

    bar_height = 0.5
    yticks, ylabels = [], []
    legend_handles: dict[str, tuple[str, str]] = {}

    for row_i, (row_label, slices) in enumerate(rows):
        y = row_i
        left = 0.0
        total = sum(s[1] for s in slices)
        for seg_label, width, color in slices:
            if width <= 0:
                continue
            ax.barh(y, width, left=left, height=bar_height, color=color,
                    edgecolor="white", linewidth=0.5)
            if total > 0 and width / total > 0.03:
                ax.text(
                    left + width / 2, y,
                    f"{width:.1f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if color not in (_C_TRANSFER, _C_DRIVER, _C_ETL, "#B8EFB0") else "#333333",
                    fontweight="bold",
                )
            left += width
            legend_handles.setdefault(seg_label, (seg_label, color))

        ax.text(
            left + (ax.get_xlim()[1] * 0.005 if ax.get_xlim()[1] > 0 else 0.5),
            y, f"{left:.1f} ms",
            ha="left", va="center", fontsize=8,
        )
        yticks.append(y)
        ylabels.append(row_label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch (cumulative timeline)")
    ax.set_title("End-to-end per-batch latency breakdown")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Legend – upper right (topology bar is short, leaving whitespace there).
    legend_patches = [
        Patch(facecolor=color, label=label, edgecolor="white")
        for label, color in legend_handles.values()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              framealpha=0.85, ncol=1)

    max_total = max(sum(s[1] for s in slices) for _, slices in rows)
    ax.set_xlim(0, max_total * 1.18)

    fig.tight_layout()
    out = (output_dir or profile_json_path.parent) / "end_to_end_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
