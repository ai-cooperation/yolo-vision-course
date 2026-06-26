"""Factory-counting rule: count tracks that cross a virtual line.

A horizontal (or vertical) line is placed at `position` (fraction of frame
height/width). Each track's centroid has a side relative to the line; when a
track flips sides, it is counted once. ByteTrack ids (from the detect layer)
are what make "count once" possible.
"""
from __future__ import annotations

from collections.abc import Iterable

from core.types import Detection, Event, FrameContext


class LineCrossingCounter:
    def __init__(self, axis: str = "horizontal", position: float = 0.5):
        if axis not in ("horizontal", "vertical"):
            raise ValueError("axis must be 'horizontal' or 'vertical'")
        self.axis = axis
        self.position = position
        self._side: dict[int, int] = {}  # track_id -> last side (-1/+1)
        self.count = 0

    def _line_px(self, ctx: FrameContext) -> float:
        return self.position * (ctx.height if self.axis == "horizontal" else ctx.width)

    def process(self, ctx: FrameContext, dets: list[Detection]) -> Iterable[Event]:
        line = self._line_px(ctx)
        events: list[Event] = []
        for d in dets:
            if d.track_id is None:
                continue
            cx, cy = d.centroid
            coord = cy if self.axis == "horizontal" else cx
            side = 1 if coord >= line else -1
            prev = self._side.get(d.track_id)
            self._side[d.track_id] = side
            if prev is not None and prev != side:
                self.count += 1
                events.append(
                    Event(
                        kind="count",
                        label=d.cls_name,
                        frame_index=ctx.index,
                        timestamp=ctx.timestamp,
                        data={
                            "track_id": d.track_id,
                            "total": self.count,
                            "direction": "down/right" if side > 0 else "up/left",
                        },
                    )
                )
        return events

    def flush(self) -> Iterable[Event]:
        return [
            Event(
                kind="count",
                label="__total__",
                frame_index=-1,
                timestamp=0.0,
                data={"total": self.count},
            )
        ]
