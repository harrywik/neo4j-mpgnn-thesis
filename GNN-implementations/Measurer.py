import csv
import time 


class Measurer:
    def __init__(self, measurements_path: str):
        self.measurements_path = measurements_path
        rows = [
        ["Event", "Time", "Value"],
        ["program_start", time.monotonic(), 1],
        ]

        # Write to a CSV file
        with open(measurements_path, "w", newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
    
    def log_event(self, event_name: str, value: int | float = 1):
        with open(self.measurements_path, "a", newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([event_name, time.monotonic(), value])
    
    def __del__(self):
        self.log_event("program_end", 1)