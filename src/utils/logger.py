from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, TextIO


class Logger:
    def __init__(self, run_dir: Path | str, metric_columns: list[str], *, flush_every: int = 10) -> None:
        if not metric_columns:
            raise ValueError("metric_columns must contain at least one column.")
        if len(set(metric_columns)) != len(metric_columns):
            raise ValueError(f"metric_columns has duplicates: {metric_columns!r}.")

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._columns: list[str] = list(metric_columns)
        self._flush_every: int = flush_every
        self._n_logged: int = 0
        self._closed: bool = False

        self._csv_path: Path = run_dir / "metrics.csv"
        self._events_path: Path = run_dir / "events.log"

        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0

        self._csv_file: TextIO = self._csv_path.open("a", newline="", encoding="utf-8")
        self._events_file: TextIO = self._events_path.open("a", encoding="utf-8")

        self._csv_writer = csv.DictWriter(self._csv_file,fieldnames=self._columns, extrasaction="ignore")

        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def log(self, **fields: Any) -> None:
        if self._closed:
            raise RuntimeError("Logger.log() called after close().")

        row: dict[str, Any] = {}
        for col in self._columns:
            val = fields.get(col, "")
            row[col] = "" if val is None else val

        self._csv_writer.writerow(row)

        parts = [f"{col}={row[col]}" for col in self._columns if row[col] != ""]
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._events_file.write(f"[{timestamp}] " + " | ".join(parts) + "\n")

        self._n_logged += 1
        if self._n_logged % self._flush_every == 0:
            self._csv_file.flush()
            self._events_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._csv_file.flush()
            self._events_file.flush()
        finally:
            self._csv_file.close()
            self._events_file.close()
            self._closed = True

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
