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

    def _find_tottime_py(file_frag: str, func_name: str) -> float:
        """Look up tottime for a Python function by filename fragment + name."""
        for (fname, _line, fn), entry in raw.items():
            if file_frag in fname and fn == func_name:
                return entry[2]
        return 0.0

    def _find_tottime_builtin(func_name_substr: str) -> float:
        """Look up tottime for a built-in / C-extension function.

        In pstats, built-in methods are stored with '~' as the filename.
        We match the function name as a substring to handle entries like
        '{method 'recv_into' of '_socket.socket' objects}' and Rust
        extensions like '{built-in method neo4j._rust.codec.packstream...}'.
        Returns the *largest* tottime among all matching entries so that
        the dominant cost is captured when multiple variants exist.
        """
        best = 0.0
        for (fname, _line, fn), entry in raw.items():
            if fname == "~" and func_name_substr in fn:
                best = max(best, entry[2])
        return best

    def _find_tottime(file_frag: str, func_name: str) -> float:
        """Prefer the built-in entry; fall back to the Python wrapper."""
        builtin = _find_tottime_builtin(func_name)
        if builtin > 0.0:
            return builtin
        return _find_tottime_py(file_frag, func_name)

    entries = [
        ("Socket receive\n(recv_into)",       "_socket",      "recv_into",   "#4C78A8"),
        ("Socket overhead\n(settimeout)",      "_socket",      "settimeout",  "#7BAFD4"),
        ("Packstream decode\n(unpack)",        "packstream",   "unpack",      "#AED4F0"),
        ("byte[] → ndarray\n(frombuffer)",     "fromnumeric",  "frombuffer",  "#9ECAE1"),
        ("Record obj. creation\n(__new__)",    "_data",        "__new__",     "#6BAED6"),
    ]

    totals_s: list[tuple[str, float, str]] = []
    for label, file_frag, func_name, color in entries:
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

    # Shared neutral colours – identical across rows so legend needs only one entry each.
    _C_DRIVER   = "#D9D9D9"   # network + Python Bolt deserialization
    _C_ETL      = "#F0F0F0"   # Python ETL

    # ── Topology slices ──────────────────────────────────────────────────────
    topo_startup     = _get("sampler_avg_query_startup_ms")
    topo_exec_ser    = _get("sampler_avg_exec_serialize_ms")
    topo_client_recv = _get("sampler_avg_client_recv_ms")
    topo_etl         = _get("topo_etl_ms")

    topo_slices: list[tuple[str, float, str]] = []
    if topo_startup is not None and topo_startup > 0:
        topo_slices.append(("DB: query startup", topo_startup, "#2171B5"))
    if topo_exec_ser is not None and topo_exec_ser > 0:
        topo_slices.append(("DB: Cypher execution + serialize", topo_exec_ser, "#4C78A8"))
    if topo_client_recv is not None and topo_client_recv > 0:
        topo_slices.append(("Network + driver recv", topo_client_recv, _C_DRIVER))
    if topo_etl is not None:
        topo_slices.append(("Python ETL", topo_etl, _C_ETL))

    # Fall back to wall-time bar when running against old profile data.
    if not topo_slices:
        topo_wall = _get("topo_fetch_ms", "sampler_avg_client_wall_time_ms")
        if topo_wall is not None:
            topo_slices.append(("Total fetch (wall)", topo_wall, "#2171B5"))
        if topo_etl is not None:
            topo_slices.append(("Python ETL", topo_etl, _C_ETL))

    # ── Feature slices ───────────────────────────────────────────────────────
    feat_startup     = _get("feat_x_avg_query_startup_ms")
    feat_exec_ser    = _get("feat_x_avg_exec_serialize_ms")
    feat_client_recv = _get("feat_x_avg_client_recv_ms")
    feat_etl         = _get("feat_x_etl_ms")

    feat_slices: list[tuple[str, float, str]] = []
    if feat_startup is not None and feat_startup > 0:
        feat_slices.append(("DB: query startup", feat_startup, "#D94F00"))
    if feat_exec_ser is not None and feat_exec_ser > 0:
        feat_slices.append(("DB: property I/O + serialize", feat_exec_ser, "#F58518"))
    if feat_client_recv is not None and feat_client_recv > 0:
        feat_slices.append(("Network + driver recv", feat_client_recv, _C_DRIVER))
    if feat_etl is not None:
        feat_slices.append(("Python ETL", feat_etl, _C_ETL))

    # Fall back to wall-time bar when running against old profile data.
    if not feat_slices:
        feat_wall = _get("feat_x_avg_client_wall_time_ms")
        if feat_wall is not None:
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
    ax.set_title("Sub-phase latency breakdown\n"
                 "DB startup | DB exec+serialize | Network+driver recv | Python ETL")
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


def plot_cpu_bar(csv_path: Path, df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Bar chart of average CPU utilization: Python/C++ vs Neo4j.

    Uses the coarse CPU samples (``python_cpu_coarse`` / ``neo4j_cpu_coarse``)
    logged every few seconds during training.  Saves as ``cpu_bar.png``.
    No-op if no coarse CPU data exists in *df*.
    """
    py_vals = pd.to_numeric(
        df.loc[df["Event"] == "python_cpu_coarse", "Value"], errors="coerce"
    ).dropna()

    neo_vals = pd.to_numeric(
        df.loc[df["Event"] == "neo4j_cpu_coarse", "Value"], errors="coerce"
    ).dropna()

    if py_vals.empty:
        return

    labels = ["Python / C++"]
    means = [float(py_vals.mean())]
    colors = ["#4C78A8"]

    if not neo_vals.empty:
        labels.append("Neo4j")
        means.append(float(neo_vals.mean()))
        colors.append("#F58518")

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.8), 4))
    bars = ax.bar(labels, means, color=colors, width=0.5)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.02,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Mean CPU utilization (%)")
    ax.set_title("Average CPU utilization by component")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = (output_dir or csv_path.parent) / "cpu_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cpu_timeline(csv_path: Path, df: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Timeline of CPU utilization from intensive per-batch burst samples.

    Plots two lines (Python/C++ and Neo4j) over wall time for the epochs
    covered by the burst monitor.  Background colour reflects the training
    phase: light blue for sampling (DB I/O), light green for ETL (Python
    assembly), light orange for GNN training.

    Uses the six event names: ``python_cpu_sampling``, ``python_cpu_etl``,
    ``python_cpu_training``, ``neo4j_cpu_sampling``, ``neo4j_cpu_etl``,
    ``neo4j_cpu_training``.

    Saves as ``cpu_timeline.png``.  No-op if no intensive CPU data exists.
    """
    py_samp  = df[df["Event"] == "python_cpu_sampling"][["Time", "Value"]].copy()
    py_etl   = df[df["Event"] == "python_cpu_etl"][["Time", "Value"]].copy()
    py_train = df[df["Event"] == "python_cpu_training"][["Time", "Value"]].copy()
    neo_samp  = df[df["Event"] == "neo4j_cpu_sampling"][["Time", "Value"]].copy()
    neo_etl   = df[df["Event"] == "neo4j_cpu_etl"][["Time", "Value"]].copy()
    neo_train = df[df["Event"] == "neo4j_cpu_training"][["Time", "Value"]].copy()

    py_all  = pd.concat([py_samp, py_etl, py_train], ignore_index=True)
    neo_all = pd.concat([neo_samp, neo_etl, neo_train], ignore_index=True)

    if py_all.empty:
        return

    # Determine t0 from first epoch or program start
    epoch_start = df.loc[df["Event"] == "epoch_start", "Time"]
    t0 = float(epoch_start.iloc[0]) if len(epoch_start) else float(df["Time"].min())

    def _prepare(frame: pd.DataFrame):
        frame = frame.copy()
        frame["rel_time"] = pd.to_numeric(frame["Time"], errors="coerce") - t0
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        frame = frame.dropna().sort_values("rel_time")
        return frame

    py_all  = _prepare(py_all)
    neo_all = _prepare(neo_all)

    # X-axis window: only the time range covered by burst samples (+5% margin)
    burst_tmin = float(py_all["rel_time"].min())
    burst_tmax = float(py_all["rel_time"].max())
    margin = max((burst_tmax - burst_tmin) * 0.05, 0.05)
    x_min = max(0.0, burst_tmin - margin)
    x_max = burst_tmax + margin

    # Absolute time bounds for filtering spans
    abs_tmin = t0 + x_min
    abs_tmax = t0 + x_max

    fig, ax = plt.subplots(figsize=(10, 4))

    # Background spans: only draw spans that overlap the burst window.
    # Render order matters: sampling (blue) first, ETL (green) on top to
    # replace its sub-window, then training (orange) in its own region.
    for phase_start_event, phase_end_event, color in [
        ("start_batch_fetch",      "end_batch_fetch",      "#FFD6D6"),
        ("start_etl",              "end_etl",              "#D4EDDA"),
        ("start_batch_processing", "end_batch_processing", "#FFF0DC"),
    ]:
        starts = df.loc[df["Event"] == phase_start_event, "Time"].to_list()
        ends   = df.loc[df["Event"] == phase_end_event,   "Time"].to_list()
        for s, e in zip(starts[:len(ends)], ends):
            s_f, e_f = float(s), float(e)
            # Skip spans entirely outside the burst window
            if e_f < abs_tmin or s_f > abs_tmax:
                continue
            ax.axvspan(s_f - t0, e_f - t0, color=color, alpha=0.45, lw=0, zorder=0)

    # Plot Python/C++ line
    ax.plot(
        py_all["rel_time"].to_numpy(),
        py_all["Value"].to_numpy(),
        color="#4C78A8", linewidth=1.2, label="Python / C++", zorder=2,
    )

    # Plot Neo4j line if data exists
    if not neo_all.empty:
        ax.plot(
            neo_all["rel_time"].to_numpy(),
            neo_all["Value"].to_numpy(),
            color="#F58518", linewidth=1.2, label="Neo4j", zorder=2,
        )

    # Legend entry for background colours
    from matplotlib.patches import Patch
    legend_handles = ax.get_legend_handles_labels()[0][:]
    legend_labels  = ax.get_legend_handles_labels()[1][:]
    legend_handles += [
        Patch(facecolor="#FFD6D6", alpha=0.8, label="DB fetching phase"),
        Patch(facecolor="#D4EDDA", alpha=0.8, label="ETL phase"),
        Patch(facecolor="#FFF0DC", alpha=0.8, label="Training phase"),
    ]
    legend_labels += ["DB fetching phase", "ETL phase", "Training phase"]

    ax.legend(handles=legend_handles, labels=legend_labels, fontsize=8, loc="upper right")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("CPU utilization (%)")
    ax.set_title("CPU utilization over time (burst samples, first N epochs)")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = (output_dir or csv_path.parent) / "cpu_timeline.png"
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

    # Colour palette for the 4-segment breakdown.
    # Segment 1: DB startup         – dark blue (topology) / dark orange (features)
    # Segment 2: DB exec+serialize  – mid blue (topology)  / orange (features)
    # Segment 3: Network+client recv – medium grey (shared)
    # Segment 4: Python ETL          – very light grey (shared)
    _C_DRIVER   = "#D9D9D9"   # network + Python Bolt deserialization
    _C_ETL      = "#F0F0F0"   # Python ETL

    # ── Topology slices ───────────────────────────────────────────────────
    # The darker sub-segment covers t_first (time to first row); the lighter
    # remainder covers t_last − t_first.  Both appear in the legend separately.
    sg = _qp_global("sampler")
    topo_startup     = sg.get("avg_query_startup_ms")
    topo_server_lat  = sg.get("avg_result_consumed_after_ms")
    topo_client_recv = sg.get("avg_client_recv_ms")
    topo_etl         = metrics.get("topo_etl_ms")

    topo_slices: list[tuple[str, float, str]] = []
    if topo_server_lat is not None and topo_server_lat > 0:
        t_first = float(topo_startup) if topo_startup and topo_startup > 0 else 0.0
        t_rest  = float(topo_server_lat) - t_first
        if t_first > 0:
            topo_slices.append((f"DB time-to-first-row ({t_first:.2f} ms)", t_first, "#08519C"))
        if t_rest > 0:
            topo_slices.append(("DB server latency (topology)", t_rest, "#2171B5"))
        else:
            topo_slices.append(("DB server latency (topology)", float(topo_server_lat), "#2171B5"))
    if topo_client_recv is not None and topo_client_recv > 0:
        topo_slices.append(("Network + driver recv", float(topo_client_recv), _C_DRIVER))
    if topo_etl is not None:
        topo_slices.append(("Python ETL", float(topo_etl), _C_ETL))

    # Fall back to wall-time bar if new metrics are absent (old profile data).
    if not topo_slices:
        topo_wall = sg.get("avg_client_wall_time_ms") or metrics.get("topo_fetch_ms")
        if topo_wall:
            topo_slices.append(("Total fetch (wall)", float(topo_wall), "#2171B5"))
        if topo_etl is not None:
            topo_slices.append(("Python ETL", float(topo_etl), _C_ETL))

    # ── Feature slices ────────────────────────────────────────────────────
    fg = _qp_global("feat_x")
    feat_startup     = fg.get("avg_query_startup_ms")
    feat_server_lat  = fg.get("avg_result_consumed_after_ms")
    feat_client_recv = fg.get("avg_client_recv_ms")
    feat_etl         = metrics.get("feat_x_etl_ms")

    feat_slices: list[tuple[str, float, str]] = []
    if feat_server_lat is not None and feat_server_lat > 0:
        t_first = float(feat_startup) if feat_startup and feat_startup > 0 else 0.0
        t_rest  = float(feat_server_lat) - t_first
        if t_first > 0:
            feat_slices.append((f"DB time-to-first-row ({t_first:.2f} ms)", t_first, "#C44E00"))
        if t_rest > 0:
            feat_slices.append(("DB server latency (features)", t_rest, "#F58518"))
        else:
            feat_slices.append(("DB server latency (features)", float(feat_server_lat), "#F58518"))
    if feat_client_recv is not None and feat_client_recv > 0:
        feat_slices.append(("Network + driver recv", float(feat_client_recv), _C_DRIVER))
    if feat_etl is not None:
        feat_slices.append(("Python ETL", float(feat_etl), _C_ETL))

    # Fall back to wall-time bar if new metrics are absent (old profile data).
    if not feat_slices:
        feat_wall = fg.get("avg_client_wall_time_ms")
        if feat_wall:
            feat_slices.append(("Total fetch (wall)", float(feat_wall), "#F58518"))
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
    fig, ax = plt.subplots(figsize=(11, max(2.5, n_rows * 1.6)))

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
            # Only label segments wide enough that the text won't crowd the
            # total-time annotation at the bar end (8% of bar = safe minimum).
            if total > 0 and width / total > 0.08:
                ax.text(
                    left + width / 2, y,
                    f"{width:.1f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if color not in (_C_DRIVER, _C_ETL, "#B8EFB0") else "#333333",
                    fontweight="bold",
                )
            left += width
            # Always overwrite so the last (lighter) shade wins for shared labels.
            legend_handles[seg_label] = (seg_label, color)

        ax.text(
            left + (ax.get_xlim()[1] * 0.03 if ax.get_xlim()[1] > 0 else 0.5),
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
    ax.set_xlim(0, max_total * 1.22)

    fig.tight_layout()
    out = (output_dir or profile_json_path.parent) / "end_to_end_latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregated multi-run plots
# ---------------------------------------------------------------------------

_DRIVER_ENTRIES = [
    ("Socket receive\n(recv_into)",     "_socket",     "recv_into",  "#4C78A8"),
    ("Socket overhead\n(settimeout)",   "_socket",     "settimeout", "#7BAFD4"),
    ("Packstream decode\n(unpack)",     "packstream",  "unpack",     "#AED4F0"),
    ("byte[] → ndarray\n(frombuffer)",  "fromnumeric", "frombuffer", "#9ECAE1"),
    ("Record obj. creation\n(__new__)", "_data",       "__new__",    "#6BAED6"),
]


def _extract_driver_ms_per_batch(prof_path: Path, n_batches: int) -> dict[str, float]:
    """Extract ms-per-batch for every tracked driver leaf from a .prof file."""
    if not prof_path.exists() or n_batches <= 0:
        return {}
    try:
        st = pstats.Stats(str(prof_path), stream=io.StringIO())
        st.strip_dirs()
    except Exception:
        return {}
    raw = st.stats  # type: ignore[attr-defined]

    def _best(file_frag: str, func_name: str) -> float:
        best = 0.0
        for (fname, _l, fn), entry in raw.items():
            if fname == "~" and func_name in fn:
                best = max(best, entry[2])
        if best > 0:
            return best
        for (fname, _l, fn), entry in raw.items():
            if file_frag in fname and fn == func_name:
                return entry[2]
        return 0.0

    return {
        label: t * 1000.0 / n_batches
        for label, file_frag, func_name, _ in _DRIVER_ENTRIES
        if (t := _best(file_frag, func_name)) > 0
    }


def _ci95(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(vals, ddof=1)) / (n ** 0.5)


def generate_avg_measurements_json(run_dirs: list[Path], output_dir: Path) -> None:
    """Write ``measurements.json`` to *output_dir* averaging all run_N subdirs.

    Scalar fields become ``{"mean": x, "std": y, "ci_95": z}``.
    Stats-object fields (dicts with ``mean_s`` key) have each sub-field averaged.
    """
    from collections import defaultdict

    def _collect(section: dict, store: dict[str, list]) -> None:
        for k, v in section.items():
            if isinstance(v, (int, float)) and v is not None:
                store[k].append(float(v))
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)) and sv is not None:
                        store[f"{k}.{sk}"].append(float(sv))

    run_scalars: dict[str, list[float]] = defaultdict(list)
    metrics_scalars: dict[str, list[float]] = defaultdict(list)

    valid_runs = 0
    for run_dir in run_dirs:
        mj = run_dir / "measurements.json"
        if not mj.exists():
            continue
        with open(mj) as f:
            data = json.load(f)
        _collect({k: v for k, v in data.get("run", {}).items() if k != "csv"}, run_scalars)
        _collect(data.get("metrics", {}), metrics_scalars)
        valid_runs += 1

    if valid_runs == 0:
        return

    def _agg(store: dict[str, list[float]]) -> dict:
        """Build nested averaged dict from flat dotted keys."""
        result: dict = {}
        for dotted_key, vals in store.items():
            parts = dotted_key.split(".", 1)
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci   = _ci95(vals)
            entry = {"mean": mean, "std": std, "ci_95": ci}
            if len(parts) == 1:
                result[parts[0]] = entry
            else:
                result.setdefault(parts[0], {})[parts[1]] = entry
        return result

    out = {
        "n_runs": valid_runs,
        "run": _agg(run_scalars),
        "metrics": _agg(metrics_scalars),
    }
    with open(output_dir / "measurements.json", "w") as f:
        json.dump(out, f, indent=2)


def plot_aggregated_folder(parent_dir: Path) -> None:
    """Generate averaged plots + ``measurements.json`` from all run_N subfolders.

    Saves to *parent_dir*:
    * ``avg_validation_convergence.png``       – epoch axis, ±95 % CI band
    * ``avg_validation_convergence_time.png``  – wall-time axis, ±95 % CI band
    * ``avg_subphase_latency_waterfall.png``   – averaged waterfall
    * ``avg_subphase_latency.png``             – bars + 95 % CI error bars
    * ``avg_driver_time_breakdown.png``        – bars + 95 % CI error bars
    * ``avg_end_to_end_latency.png``           – averaged from query_profile.json
    * ``avg_cpu_bar.png``                      – CPU bar + 95 % CI error bars
    * ``avg_phase_summary.png``                – phase bar + 95 % CI error bars
    * ``measurements.json``                    – averaged scalar metrics
    """
    from collections import defaultdict

    run_dirs = sorted(
        [p for p in parent_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    if not run_dirs:
        return

    # ------------------------------------------------------------------
    # 1. Collect per-run scalar metrics and batch counts
    # ------------------------------------------------------------------
    metric_vals: dict[str, list[float]] = defaultdict(list)
    n_batches_list: list[int] = []

    for run_dir in run_dirs:
        mj = run_dir / "measurements.json"
        if not mj.exists():
            continue
        with open(mj) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        run_info = data.get("run", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                metric_vals[k].append(float(v))
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)) and sv is not None:
                        metric_vals[f"{k}.{sk}"].append(float(sv))
        nb = run_info.get("batches_seen")
        if nb is not None:
            n_batches_list.append(int(nb))

    if not metric_vals:
        return

    avg_metrics = {k: float(np.mean(v)) for k, v in metric_vals.items()}
    std_metrics = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in metric_vals.items()}
    n_runs = len(run_dirs)
    dummy_csv = parent_dir / "_avg_dummy.csv"

    # ------------------------------------------------------------------
    # 2. averaged measurements.json
    # ------------------------------------------------------------------
    generate_avg_measurements_json(run_dirs, parent_dir)

    # ------------------------------------------------------------------
    # 3. Subphase waterfall (mean only – CI not meaningful on stacked)
    # ------------------------------------------------------------------
    plot_subphase_latency_waterfall(dummy_csv, {"metrics": avg_metrics}, output_dir=parent_dir)
    _safe_rename(parent_dir / "subphase_latency_waterfall.png",
                 parent_dir / "avg_subphase_latency_waterfall.png")

    # ------------------------------------------------------------------
    # 4. Driver time breakdown bars + CI error bars
    # ------------------------------------------------------------------
    all_driver: dict[str, list[float]] = defaultdict(list)
    for run_dir, nb in zip(run_dirs, n_batches_list or [1] * len(run_dirs)):
        for label, v in _extract_driver_ms_per_batch(run_dir / "train_profile.prof", nb).items():
            all_driver[label].append(v)
    if all_driver:
        _plot_avg_driver_breakdown_ci(all_driver, parent_dir)

    # ------------------------------------------------------------------
    # 6. Validation convergence vs epochs (±95 % CI band)
    # ------------------------------------------------------------------
    all_acc: list[np.ndarray] = []
    for run_dir in run_dirs:
        vp = run_dir / "validation_accuracies.csv"
        if not vp.exists():
            continue
        try:
            df = pd.read_csv(vp)
            vals = pd.to_numeric(df["Value"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals):
                all_acc.append(vals)
        except Exception:
            pass
    if all_acc:
        min_len = min(len(s) for s in all_acc)
        stacked = np.stack([s[:min_len] for s in all_acc], axis=0)
        mean_acc = stacked.mean(axis=0)
        ci_acc   = 1.96 * stacked.std(axis=0, ddof=1) / (len(all_acc) ** 0.5) if len(all_acc) > 1 else np.zeros_like(mean_acc)
        _plot_avg_val_convergence_epochs(mean_acc, ci_acc, parent_dir)

    # ------------------------------------------------------------------
    # 7. Validation convergence vs wall time (±95 % CI band)
    # ------------------------------------------------------------------
    _plot_avg_val_convergence_time(run_dirs, parent_dir)

    # ------------------------------------------------------------------
    # 8. End-to-end latency waterfall averaged from query_profile.json
    # ------------------------------------------------------------------
    _plot_avg_end_to_end_latency(run_dirs, avg_metrics, parent_dir)

    # ------------------------------------------------------------------
    # 9. CPU bar + CI error bars
    # ------------------------------------------------------------------
    _plot_avg_cpu_bar_ci(run_dirs, parent_dir)

    # ------------------------------------------------------------------
    # 10. Phase summary bar + CI error bars
    # ------------------------------------------------------------------
    _plot_avg_phase_summary_ci(metric_vals, std_metrics, n_runs, parent_dir)

    # ------------------------------------------------------------------
    # 11. CPU timeline averaged across runs
    # ------------------------------------------------------------------
    _plot_avg_cpu_timeline(run_dirs, parent_dir)

    print(f"  Averaged plots written to: {parent_dir}")


# ── helpers ─────────────────────────────────────────────────────────────────

def _safe_rename(src: Path, dst: Path) -> None:
    if src.exists():
        src.replace(dst)


def _ci_band(vals_list: np.ndarray) -> np.ndarray:
    """95 % CI half-width (array)."""
    n = vals_list.shape[0]
    if n < 2:
        return np.zeros(vals_list.shape[1])
    return 1.96 * vals_list.std(axis=0, ddof=1) / (n ** 0.5)


def _plot_avg_subphase_latency_ci(
    avg_metrics: dict[str, float],
    std_metrics: dict[str, float],
    n_runs: int,
    output_dir: Path,
) -> None:
    topo_segments = [
        ("topo_fetch_ms",               "Topology: total fetch (wall)",         "#2171B5"),
        ("sampler_avg_db_exec_time_ms", "Topology: DB Cypher execution",        "#4C78A8"),
        ("network_baseline_ms",         "Topology: transfer + driver overhead", "#7BAFD4"),
        ("topo_etl_ms",                 "Topology: Python ETL",                 "#AED4F0"),
    ]
    feat_segments = [
        ("feat_x_avg_client_wall_time_ms", "Features: total fetch",             "#D94F00"),
        ("feat_x_avg_db_exec_time_ms",     "Features: DB property I/O",         "#F58518"),
        ("feat_x_avg_driver_overhead_ms",  "Features: transfer + driver",       "#F7A850"),
        ("feat_x_etl_ms",                  "Features: Python ETL",              "#FAC980"),
    ]

    rows = []
    for key, label, color in topo_segments + feat_segments:
        v = avg_metrics.get(key)
        if v is not None:
            std = std_metrics.get(key, 0.0)
            ci = 1.96 * std / (n_runs ** 0.5) if n_runs > 1 else 0.0
            rows.append((label, float(v), ci, color))

    if not rows:
        return

    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    cis    = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, xerr=cis, color=colors,
                   error_kw={"ecolor": "#333333", "capsize": 3, "linewidth": 1})

    for bar, val, ci in zip(bars, values, cis):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} ± {ci:.2f} ms",
                va="center", ha="left", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch (± 95 % CI)")
    ax.set_title("Sub-phase latency – averaged across runs")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_subphase_latency.png", dpi=150)
    plt.close(fig)


def _plot_avg_driver_breakdown_ci(
    all_driver: dict[str, list[float]],
    output_dir: Path,
) -> None:
    color_map = {e[0]: e[3] for e in _DRIVER_ENTRIES}
    items = [(lbl, float(np.mean(v)), _ci95(v)) for lbl, v in all_driver.items() if v]
    if not items:
        return

    total_mean = sum(m for _, m, _ in items)
    labels  = [l for l, _, _ in items]
    means   = [m for _, m, _ in items]
    cis     = [c for _, _, c in items]
    colors  = [color_map.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(items) * 2), 3.2))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, means, xerr=cis, color=colors,
            error_kw={"ecolor": "#333333", "capsize": 3, "linewidth": 1})

    for i, (val, ci) in enumerate(zip(means, cis)):
        ax.text(val + max(means) * 0.01, i,
                f"{val:.2f} ± {ci:.2f} ms",
                va="center", ha="left", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.replace("\n", " ") for l in labels], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch (± 95 % CI)")
    ax.set_title(f"Driver time breakdown – averaged across runs  (total {total_mean:.2f} ms)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_driver_time_breakdown.png", dpi=150)
    plt.close(fig)


def _plot_avg_val_convergence_epochs(
    mean_acc: np.ndarray,
    ci_acc: np.ndarray,
    output_dir: Path,
) -> None:
    epochs = np.arange(1, len(mean_acc) + 1, dtype=float)
    color = "#E45756"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, mean_acc, color=color, linewidth=1.8, label="mean")
    ax.fill_between(epochs,
                    np.clip(mean_acc - ci_acc, 0, 1),
                    np.clip(mean_acc + ci_acc, 0, 1),
                    alpha=0.25, color=color, label="95 % CI")
    best_idx = int(np.argmax(mean_acc))
    ax.axvline(x=float(epochs[best_idx]), color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(float(epochs[best_idx]), 0.02, f"epoch {int(epochs[best_idx])}",
            color=color, fontsize=7, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_title("Validation accuracy vs epochs – averaged across runs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_validation_convergence.png", dpi=150)
    plt.close(fig)


def _plot_avg_val_convergence_time(run_dirs: list[Path], output_dir: Path) -> None:
    """Averaged validation accuracy vs elapsed wall time, ±95 % CI band."""
    from .measurements_summary import read_measurements

    series: list[tuple[np.ndarray, np.ndarray]] = []  # (times, accs) per run
    for run_dir in run_dirs:
        csv_p = run_dir / "measurements.csv"
        val_p = run_dir / "validation_accuracies.csv"
        if not csv_p.exists() or not val_p.exists():
            continue
        try:
            df = read_measurements(csv_p)
            epoch_starts = df.loc[df["Event"] == "epoch_start", "Time"]
            t0 = float(epoch_starts.iloc[0]) if len(epoch_starts) else None
            if t0 is None:
                continue
            vdf = pd.read_csv(val_p)
            times = pd.to_numeric(vdf["Time"], errors="coerce").dropna().to_numpy(dtype=float) - t0
            accs  = pd.to_numeric(vdf["Value"], errors="coerce").dropna().to_numpy(dtype=float)
            n = min(len(times), len(accs))
            if n > 0:
                series.append((times[:n], accs[:n]))
        except Exception:
            pass

    if not series:
        return

    # Interpolate each run onto a common time grid
    max_t = min(s[0][-1] for s in series)  # only up to min of all max times
    grid = np.linspace(0, max_t, 200)
    interp_accs = []
    for times, accs in series:
        interp_accs.append(np.interp(grid, times, accs))

    stacked   = np.stack(interp_accs, axis=0)
    mean_acc  = stacked.mean(axis=0)
    ci_acc    = _ci_band(stacked)

    color = "#E45756"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mean_acc, color=color, linewidth=1.8, label="mean")
    ax.fill_between(grid,
                    np.clip(mean_acc - ci_acc, 0, 1),
                    np.clip(mean_acc + ci_acc, 0, 1),
                    alpha=0.25, color=color, label="95 % CI")
    best_idx = int(np.argmax(mean_acc))
    ax.axvline(x=grid[best_idx], color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(grid[best_idx], 0.02, f"{grid[best_idx]:.1f}s",
            color=color, fontsize=7, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_title("Validation accuracy vs wall time – averaged across runs")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_validation_convergence_time.png", dpi=150)
    plt.close(fig)


def _plot_avg_end_to_end_latency(
    run_dirs: list[Path],
    avg_metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Averaged end-to-end latency waterfall from query_profile.json files."""
    from collections import defaultdict
    sampler_vals: dict[str, list[float]] = defaultdict(list)
    feat_vals: dict[str, list[float]] = defaultdict(list)

    for run_dir in run_dirs:
        qp_path = run_dir / "query_profile.json"
        if not qp_path.exists():
            continue
        with open(qp_path) as f:
            qp = json.load(f)
        for k, v in qp.get("sampler", {}).get("global", {}).items():
            if isinstance(v, (int, float)):
                sampler_vals[k].append(float(v))
        for k, v in qp.get("feat_x", {}).get("global", {}).items():
            if isinstance(v, (int, float)):
                feat_vals[k].append(float(v))

    if not sampler_vals and not feat_vals:
        return

    sg = {k: float(np.mean(v)) for k, v in sampler_vals.items()}
    fg = {k: float(np.mean(v)) for k, v in feat_vals.items()}

    _C_DRIVER = "#D9D9D9"
    _C_ETL    = "#F0F0F0"

    topo_slices: list[tuple[str, float, str]] = []
    topo_server = sg.get("avg_result_consumed_after_ms")
    topo_startup = sg.get("avg_query_startup_ms", 0.0)
    topo_recv = sg.get("avg_client_recv_ms")
    topo_etl = avg_metrics.get("topo_etl_ms")
    if topo_server and topo_server > 0:
        t_first = float(topo_startup) if topo_startup else 0.0
        t_rest  = float(topo_server) - t_first
        if t_first > 0:
            topo_slices.append((f"DB time-to-first-row", t_first, "#08519C"))
        if t_rest > 0:
            topo_slices.append(("DB server latency (topology)", t_rest, "#2171B5"))
    if topo_recv and topo_recv > 0:
        topo_slices.append(("Network + driver recv", float(topo_recv), _C_DRIVER))
    if topo_etl:
        topo_slices.append(("Python ETL", float(topo_etl), _C_ETL))

    feat_slices: list[tuple[str, float, str]] = []
    feat_server = fg.get("avg_result_consumed_after_ms")
    feat_startup = fg.get("avg_query_startup_ms", 0.0)
    feat_recv = fg.get("avg_client_recv_ms")
    feat_etl = avg_metrics.get("feat_x_etl_ms")
    if feat_server and feat_server > 0:
        t_first = float(feat_startup) if feat_startup else 0.0
        t_rest  = float(feat_server) - t_first
        if t_first > 0:
            feat_slices.append(("DB time-to-first-row", t_first, "#C44E00"))
        if t_rest > 0:
            feat_slices.append(("DB server latency (features)", t_rest, "#F58518"))
    if feat_recv and feat_recv > 0:
        feat_slices.append(("Network + driver recv", float(feat_recv), _C_DRIVER))
    if feat_etl:
        feat_slices.append(("Python ETL", float(feat_etl), _C_ETL))

    rows = []
    if topo_slices:
        rows.append(("Topology", topo_slices))
    if feat_slices:
        rows.append(("Features", feat_slices))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(11, max(2.5, len(rows) * 1.6)))
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
            if total > 0 and width / total > 0.08:
                ax.text(left + width / 2, y, f"{width:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if color not in (_C_DRIVER, _C_ETL) else "#333333",
                        fontweight="bold")
            left += width
            legend_handles[seg_label] = (seg_label, color)
        ax.text(left + (ax.get_xlim()[1] * 0.03 if ax.get_xlim()[1] > 0 else 0.5),
                y, f"{left:.1f} ms", ha="left", va="center", fontsize=8)
        yticks.append(y)
        ylabels.append(row_label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch (avg across runs)")
    ax.set_title("End-to-end latency breakdown – averaged across runs")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    max_total = max(sum(s[1] for s in slices) for _, slices in rows)
    ax.set_xlim(0, max_total * 1.22)
    legend_patches = [Patch(facecolor=c, label=l, edgecolor="white")
                      for l, c in legend_handles.values()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
              framealpha=0.85, ncol=1)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_end_to_end_latency.png", dpi=150)
    plt.close(fig)


def _plot_avg_cpu_bar_ci(run_dirs: list[Path], output_dir: Path) -> None:
    """Averaged CPU bar (Python vs Neo4j) with 95 % CI error bars."""
    from .measurements_summary import read_measurements
    py_vals, neo_vals = [], []
    for run_dir in run_dirs:
        csv_p = run_dir / "measurements.csv"
        if not csv_p.exists():
            continue
        try:
            df = read_measurements(csv_p)
            py = pd.to_numeric(df.loc[df["Event"] == "python_cpu_coarse", "Value"],
                               errors="coerce").dropna()
            neo = pd.to_numeric(df.loc[df["Event"] == "neo4j_cpu_coarse", "Value"],
                                errors="coerce").dropna()
            if not py.empty:
                py_vals.append(float(py.mean()))
            if not neo.empty:
                neo_vals.append(float(neo.mean()))
        except Exception:
            pass

    if not py_vals:
        return

    labels = ["Python / C++"]
    means  = [float(np.mean(py_vals))]
    cis    = [_ci95(py_vals)]
    colors = ["#4C78A8"]
    if neo_vals:
        labels.append("Neo4j")
        means.append(float(np.mean(neo_vals)))
        cis.append(_ci95(neo_vals))
        colors.append("#F58518")

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.8), 4))
    bars = ax.bar(labels, means, yerr=cis, color=colors, width=0.5,
                  error_kw={"ecolor": "#333333", "capsize": 5, "linewidth": 1.2})
    for bar, val, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.02,
                f"{val:.1f} ± {ci:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean CPU utilization (%)")
    ax.set_title("Average CPU utilization – averaged across runs")
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_cpu_bar.png", dpi=150)
    plt.close(fig)


def _plot_avg_cpu_timeline(run_dirs: list[Path], output_dir: Path) -> None:
    """Averaged CPU utilization timeline (Python vs Neo4j) with ±95 % CI band."""
    from .measurements_summary import read_measurements

    py_series: list[tuple[np.ndarray, np.ndarray]] = []
    neo_series: list[tuple[np.ndarray, np.ndarray]] = []

    for run_dir in run_dirs:
        csv_p = run_dir / "measurements.csv"
        if not csv_p.exists():
            continue
        try:
            df = read_measurements(csv_p)
            epoch_start = df.loc[df["Event"] == "epoch_start", "Time"]
            t0 = float(epoch_start.iloc[0]) if len(epoch_start) else float(df["Time"].min())

            def _rel(events):
                sub = df[df["Event"].isin(events)][["Time", "Value"]].copy()
                sub["t"] = pd.to_numeric(sub["Time"], errors="coerce") - t0
                sub["v"] = pd.to_numeric(sub["Value"], errors="coerce")
                sub = sub.dropna().sort_values("t")
                return sub["t"].to_numpy(dtype=float), sub["v"].to_numpy(dtype=float)

            py_t, py_v = _rel(["python_cpu_sampling", "python_cpu_etl", "python_cpu_training"])
            neo_t, neo_v = _rel(["neo4j_cpu_sampling", "neo4j_cpu_etl", "neo4j_cpu_training"])
            if len(py_t) > 1:
                py_series.append((py_t, py_v))
            if len(neo_t) > 1:
                neo_series.append((neo_t, neo_v))
        except Exception:
            pass

    if not py_series:
        return

    max_t = min(s[0][-1] for s in py_series)
    grid = np.linspace(0.0, max_t, 300)

    def _interp_ci(series):
        stacked = np.stack([np.interp(grid, t, v) for t, v in series], axis=0)
        mean = stacked.mean(axis=0)
        ci = _ci_band(stacked)
        return mean, ci

    py_mean, py_ci = _interp_ci(py_series)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, py_mean, color="#4C78A8", linewidth=1.4, label="Python / C++")
    ax.fill_between(grid, np.clip(py_mean - py_ci, 0, None),
                    py_mean + py_ci, alpha=0.25, color="#4C78A8")

    if neo_series:
        neo_mean, neo_ci = _interp_ci(neo_series)
        ax.plot(grid, neo_mean, color="#F58518", linewidth=1.4, label="Neo4j")
        ax.fill_between(grid, np.clip(neo_mean - neo_ci, 0, None),
                        neo_mean + neo_ci, alpha=0.25, color="#F58518")

    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("CPU utilization (%)")
    ax.set_title("CPU utilization over time – averaged across runs (±95 % CI)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_cpu_timeline.png", dpi=150)
    plt.close(fig)


def _plot_avg_phase_summary_ci(
    metric_vals: dict[str, list[float]],
    std_metrics: dict[str, float],
    n_runs: int,
    output_dir: Path,
) -> None:
    """Averaged phase summary (sampling vs training) with 95 % CI error bars."""
    s_key = "sampling_phase_time_s.mean_s"
    t_key = "training_phase_time_s.mean_s"
    sv = metric_vals.get(s_key, [])
    tv = metric_vals.get(t_key, [])
    if not sv and not tv:
        return

    labels = []
    means  = []
    cis    = []
    colors = []
    if sv:
        labels.append("Sampling")
        means.append(float(np.mean(sv)))
        cis.append(_ci95(sv))
        colors.append("#4C78A8")
    if tv:
        labels.append("Training")
        means.append(float(np.mean(tv)))
        cis.append(_ci95(tv))
        colors.append("#F58518")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=cis, color=colors,
                  error_kw={"ecolor": "#333333", "capsize": 5, "linewidth": 1.2})
    for bar, val, ci in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(means) * 0.02,
                f"{val*1000:.2f} ± {ci*1000:.2f} ms",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean duration per batch (s)")
    ax.set_title("Phase durations – averaged across runs")
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_phase_summary.png", dpi=150)
    plt.close(fig)
