"""Immutable data objects passed between pipeline stages.

Frozen dataclasses: stages return new objects, never mutate inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """A single object detection, optionally with a tracker id."""

    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # full-frame pixel coords
    track_id: int | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True)
class FrameContext:
    """Metadata about the frame currently flowing through the pipeline."""

    index: int
    timestamp: float
    width: int
    height: int
    motion: bool = True  # did the gate see motion this frame


@dataclass(frozen=True)
class Event:
    """An output emitted by a rule, consumed by sinks."""

    kind: str  # "alert" | "log" | "count"
    label: str
    frame_index: int
    timestamp: float
    data: dict = field(default_factory=dict)
