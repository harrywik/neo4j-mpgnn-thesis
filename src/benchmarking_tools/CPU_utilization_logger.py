import psutil
import threading


def find_neo4j_process() -> psutil.Process | None:
    """Scan the process table for the Neo4j JVM process.

    Returns a psutil.Process handle for the first process whose command line
    contains 'org.neo4j', or None if Neo4j is not running.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "org.neo4j" in cmdline or "com.neo4j" in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def _sample_process_cpu(proc: psutil.Process) -> float:
    """Return CPU % for proc plus all its live children (non-blocking)."""
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


def start_cpu_burst(measurer):
    """Start an intensive 10 ms CPU burst monitor for a single batch.

    Samples Python and Neo4j CPU every 10 ms.  The current training phase is
    read from ``measurer.get_phase()`` on each tick to select the event name:

    * ``python_cpu_sampling`` / ``neo4j_cpu_sampling``  (phase == "sampling")
    * ``python_cpu_training`` / ``neo4j_cpu_training``  (phase == "training")

    ``measurer.neo4j_proc`` must already be set (done by ``start_cpu_monitor``).
    Call before the batch starts; call ``stop_event.set(); thread.join()`` after.

    Returns ``(stop_event, thread)``.
    """
    python_proc = psutil.Process()
    neo4j_proc = getattr(measurer, "neo4j_proc", None)

    # Warm up handles for this burst window.
    python_proc.cpu_percent(interval=None)
    if neo4j_proc is not None:
        try:
            neo4j_proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            neo4j_proc = None

    stop_event = threading.Event()

    def burst():
        while not stop_event.is_set():
            phase = measurer.get_phase()
            if phase == "sampling":
                suffix = "sampling"
            elif phase == "etl":
                suffix = "etl"
            else:
                suffix = "training"

            py_cpu = _sample_process_cpu(python_proc)
            measurer.log_event(f"python_cpu_{suffix}", py_cpu)

            if neo4j_proc is not None:
                try:
                    neo4j_cpu = neo4j_proc.cpu_percent(interval=None)
                    measurer.log_event(f"neo4j_cpu_{suffix}", neo4j_cpu)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            stop_event.wait(0.0001)  # 0.5 ms

    t = threading.Thread(target=burst, daemon=True)
    t.start()
    return stop_event, t
