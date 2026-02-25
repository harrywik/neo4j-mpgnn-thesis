from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ConfigWriter:
    def __init__(self, run_results_path: Path, config: Dict[str, Any]) -> None:
        self.config_path = run_results_path / "config.json"
        self.config_data: Dict[str, Any] = dict(config)
        self._write()

    def update(self, key: str, value: Any) -> None:
        self.config_data[key] = value
        self._write()

    def _write(self) -> None:
        with open(self.config_path, "w") as f:
            json.dump(self.config_data, f, indent=4)
