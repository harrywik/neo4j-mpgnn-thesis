import psutil
import threading
import time

def start_cpu_monitor(measurer, interval=1):
    """Measures 100% as one CPU core, computer with several cores can have more
    than 100% utilization"""
    if interval is None or interval <= 0:
        return None
    process = psutil.Process()
    process.cpu_percent(1)  # warmup

    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            procs = [process] + process.children(recursive=True)
            cpu = 0.0
            for p in procs:
                try:
                    cpu += p.cpu_percent(None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # RSS memory in MB
            try:
                mem_bytes = sum(p.memory_info().rss for p in procs)
                mem_mb = mem_bytes / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                mem_mb = 0.0

            measurer.log_event("cpu_utilization_percentage", cpu)
            measurer.log_event("ram_usage_mb", mem_mb)
            time.sleep(interval)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return stop_event, t