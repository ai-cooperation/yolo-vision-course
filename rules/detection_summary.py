"""Teaching rule: accumulate detections and emit a recognition report.

This is what makes the teaching tier's "辨識驗證" observable: per class it
reports total detections (all frames) and unique tracked objects (ByteTrack
ids). Emitted once at stream end via flush().
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from core.types import Detection, Event, FrameContext


class DetectionSummary:
    def __init__(self):
        self._counts: Counter = Counter()
        self._tracks: dict[str, set] = {}

    def process(self, ctx: FrameContext, dets: list[Detection]) -> Iterable[Event]:
        for d in dets:
            self._counts[d.cls_name] += 1
            if d.track_id is not None:
                self._tracks.setdefault(d.cls_name, set()).add(d.track_id)
        return []

    def flush(self) -> Iterable[Event]:
        events: list[Event] = []
        for name, n in self._counts.most_common():
            events.append(
                Event(
                    kind="report",
                    label=name,
                    frame_index=-1,
                    timestamp=0.0,
                    data={"detections": n, "unique_tracks": len(self._tracks.get(name, ()))},
                )
            )
        return events
