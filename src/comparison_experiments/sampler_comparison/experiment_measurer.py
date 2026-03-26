from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path

from benchmarking_tools.Measurer import Measurer
from benchmarking_tools.config_writer import ConfigWriter


class ExperimentMeasurer(Measurer):
    """Measurer variant that writes to an explicit directory.

    The base :class:`Measurer` auto-discovers the next ``run_N`` folder under
    ``experiment_results/results/``.  ``ExperimentMeasurer`` accepts a fully
    qualified ``run_dir`` instead, so the sampler-comparison orchestrator can
    place each run under its own sampler subfolder without polluting the global
    run counter.

    All methods (``log_event``, ``summarize``, ``close``, ``write_to_configresult``,
    ``__del__``) are inherited unchanged from ``Measurer``.
    """

    def __init__(self, run_dir: Path, config: dict) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

        self.config_writer = ConfigWriter(run_dir, config)
        self.measurements_path = run_dir / "measurements.csv"
        self.run_results_path = run_dir
        self.profile_accumulator = None
        self.node_visit_counter: Counter = Counter()
        self.edge_visit_counter: Counter = Counter()

        self._csvfile = open(self.measurements_path, "w", newline="\n")
        self._writer = csv.writer(self._csvfile)
        self._writer.writerow(["Event", "Time", "Value"])

        self.log_event("program_start", 1)
