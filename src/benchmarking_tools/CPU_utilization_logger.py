import psutil
import threading
import time

def start_cpu_monitor(measurer, interval=1):
    """Measures 100% as one CPU core, computer with several cores can have more
    than 100% utilization"""
    if interval is None or interval <= 0:
        return None

    # Warm up the system-wide CPU counter so the first real sample is valid.
    psutil.cpu_percent(interval=None)
    process = psutil.Process()

    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            # System-wide CPU percent (sum across all cores, non-blocking).
            cpu = psutil.cpu_percent(interval=None)

            # RSS of the current process only — no child scan needed.
            try:
                mem_mb = process.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                mem_mb = 0.0

            measurer.log_event("cpu_utilization_percentage", cpu)
            measurer.log_event("ram_usage_mb", mem_mb)
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return stop_event, t