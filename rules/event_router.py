"""Home-sensing rule: classify and route detections into events.

person/car -> "alert" (store image + notify, in real sinks)
animals     -> "log" (metadata only)
everything else / no detection -> dropped as ambient noise

Alerts are de-duplicated by track_id so one subject crossing the frame fires
once, not once per frame.
"""
from __future__ import annotations

from collections.abc import Iterable

from core.types import Detection, Event, FrameContext


class EventRouter:
    def __init__(self, alert_classes=None, log_classes=None):
        self.alert_classes = set(alert_classes or ["person", "car"])
        self.log_classes = set(log_classes or ["bird", "cat", "dog"])
        self._seen_alert: set[tuple[int, str]] = set()

    def process(self, ctx: FrameContext, dets: list[Detection]) -> Iterable[Event]:
        events: list[Event] = []
        for d in dets:
            if d.cls_name in self.alert_classes:
                key = (d.track_id if d.track_id is not None else -1, d.cls_name)
                if key in self._seen_alert:
                    continue
                self._seen_alert.add(key)
                events.append(
                    Event(
                        kind="alert",
                        label=d.cls_name,
                        frame_index=ctx.index,
                        timestamp=ctx.timestamp,
                        data={"track_id": d.track_id, "conf": round(d.conf, 3)},
                    )
                )
            elif d.cls_name in self.log_classes:
                events.append(
                    Event(
                        kind="log",
                        label=d.cls_name,
                        frame_index=ctx.index,
                        timestamp=ctx.timestamp,
                        data={"track_id": d.track_id, "conf": round(d.conf, 3)},
                    )
                )
        return events

    def flush(self) -> Iterable[Event]:
        return []
