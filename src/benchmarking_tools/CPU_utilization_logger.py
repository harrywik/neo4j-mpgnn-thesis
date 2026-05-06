import math
import threading
import time

import numpy as np
import psutil


def find_neo4j_process() -> psutil.Process | None:
    """Scan the process table for the Neo4j JVM process.

    On some Linux packaging (e.g. Debian/apt on GCP), Neo4j is started by a
    lightweight NeoBoot launcher (org.neo4j.server.startup.NeoBoot, -Xmx128m)
    which then spawns the real database JVM
    (com.neo4j.server.enterprise.Neo4jEnterprise, -Xms/-Xmx=32 GB+).
    Returning the launcher gives ~0 % CPU readings.

    Strategy: collect all candidate processes and return the one with the
    largest RSS — the real server will always dwarf the bootstrap wrapper.
    """
    candidates: list[tuple[int, psutil.Process]] = []  # (rss_bytes, proc)
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "org.neo4j" in cmdline or "com.neo4j" in cmdline:
                try:
                    rss = proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss = 0
                candidates.append((rss, proc))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _sample_process_cpu(proc: psutil.Process) -> float:
    """Return CPU % for proc plus all its live children (non-blocking).

    Used only by the coarse monitor which runs at a 5 s interval where
    cpu_percent is reliable. The burst monitor uses cumulative cpu_times
    instead.
    """
    try:
        total = proc.cpu_percent(interval=None)
        for child in proc.children(recursive=True):
            try:
                total += child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def start_cpu_monitor(measurer, interval: float = 5):
    """Start a background coarse CPU monitor thread.

    Samples the Python training process (including child workers) and the
    Neo4j JVM process (if running) every *interval* seconds.  Logs the events
    ``python_cpu_coarse`` and ``neo4j_cpu_coarse`` via *measurer*.

    Also stores the found Neo4j process handle on ``measurer.neo4j_proc`` so
    that ``start_cpu_burst`` can reuse it without re-scanning.

    Returns ``(stop_event, thread)`` or ``None`` if interval <= 0.
    """
    if interval is None or interval <= 0:
        return None

    python_proc = psutil.Process()
    neo4j_proc = find_neo4j_process()
    measurer.neo4j_proc = neo4j_proc

    # Warm up both handles so the first real sample is valid.
    python_proc.cpu_percent(interval=None)
    if neo4j_proc is not None:
        try:
            neo4j_proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            measurer.neo4j_proc = None
            neo4j_proc = None

    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            py_cpu = _sample_process_cpu(python_proc)
            measurer.log_event("python_cpu_coarse", py_cpu)

            if neo4j_proc is not None:
                try:
                    neo4j_cpu = neo4j_proc.cpu_percent(interval=None)
                    measurer.log_event("neo4j_cpu_coarse", neo4j_cpu)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            mem_mb = 0.0
            try:
                mem_mb = python_proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            measurer.log_event("ram_usage_mb", mem_mb)

            stop_event.wait(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return stop_event, t


def start_cpu_burst(measurer, period_s: float = 0.001, max_samples: int = 8192):
    """Start an intensive burst CPU monitor for the first N batches.

    Records *cumulative* CPU time (seconds) for Python and Neo4j once every
    *period_s* seconds into pre-allocated numpy arrays, then flushes all rows
    to the measurements CSV in a single batch when the burst stops.

    Event names written:
    * ``python_cpu_time_s`` — cumulative CPU-seconds of the Python process
      (all threads, via ``time.process_time()``; no child-process walk).
    * ``neo4j_cpu_time_s``  — cumulative user+system CPU-seconds of the Neo4j
      JVM process (via ``psutil.Process.cpu_times()``).

    The plotter derives CPU % offline from consecutive Δcpu / Δwall pairs,
    which eliminates the 10 ms jiffy-quantisation artefacts that the old
    per-tick ``cpu_percent`` approach produced.

    ``measurer.neo4j_proc`` must already be set (done by ``start_cpu_monitor``).
    Call before the first burst batch starts; call
    ``stop_event.set(); thread.join()`` after the last burst batch ends.

    Returns ``(stop_event, thread)``.
    """
    neo4j_proc = getattr(measurer, "neo4j_proc", None)

    # Warm up the Neo4j cpu_times handle so the first delta is valid.
    if neo4j_proc is not None:
        try:
            neo4j_proc.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            neo4j_proc = None

    buf_t   = np.empty(max_samples, dtype=np.float64)
    buf_py  = np.empty(max_samples, dtype=np.float64)
    buf_neo = np.full(max_samples, math.nan, dtype=np.float64)

    stop_event = threading.Event()

    def burst():
        i = 0
        while not stop_event.is_set() and i < max_samples:
            buf_t[i]  = time.monotonic()
            buf_py[i] = time.process_time()
            if neo4j_proc is not None:
                try:
                    ct = neo4j_proc.cpu_times()
                    buf_neo[i] = ct.user + ct.system
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            i += 1
            stop_event.wait(period_s)

        # Flush all buffered samples in one batch after the burst window ends.
        # Using explicit timestamps so the values reflect when each sample was
        # taken, not when the flush happens.
        for k in range(i):
            measurer.log_event("python_cpu_time_s", buf_py[k], t=buf_t[k])
            if not math.isnan(buf_neo[k]):
                measurer.log_event("neo4j_cpu_time_s", buf_neo[k], t=buf_t[k])

    t = threading.Thread(target=burst, daemon=True)
    t.start()
    return stop_event, t
