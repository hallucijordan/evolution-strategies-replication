"""
Lightweight CSV logger for training runs.
Each call to log() appends one row; headers are written on the first call.
"""

import csv
import time
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str, run_name: str):
        self.run_dir = Path(log_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "training_log.csv"
        self._fields: list[str] | None = None
        self._start = time.time()

    # ------------------------------------------------------------------
    def log(self, **kwargs) -> None:
        """Append one row.  First call determines the column order."""
        if self._fields is None:
            self._fields = list(kwargs.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(kwargs)

        elapsed = time.time() - self._start
        parts = []
        for k, v in kwargs.items():
            parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
        print(f"[{elapsed:6.0f}s] " + " | ".join(parts))

    # ------------------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        """Convenience: directory for saving model checkpoints."""
        d = self.run_dir / "checkpoints"
        d.mkdir(exist_ok=True)
        return d


# ---------------------------------------------------------------------------

def load_results(log_dir: str, run_name: str):
    """Load a training log CSV as a pandas DataFrame."""
    import pandas as pd
    csv_path = Path(log_dir) / run_name / "training_log.csv"
    return pd.read_csv(csv_path)
