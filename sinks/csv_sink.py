"""CSV sink: append events to a CSV file (works for both profiles)."""
from __future__ import annotations

import csv
import json

from core.types import Event


class CsvSink:
    FIELDS = ["kind", "label", "frame_index", "timestamp", "data"]

    def __init__(self, path: str = "events.csv"):
        self.path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def emit(self, event: Event) -> None:
        self._writer.writerow(
            {
                "kind": event.kind,
                "label": event.label,
                "frame_index": event.frame_index,
                "timestamp": round(event.timestamp, 3),
                "data": json.dumps(event.data, ensure_ascii=False),
            }
        )

    def close(self) -> None:
        self._fh.close()
