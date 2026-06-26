"""Rule protocol (pluggable slot #2).

A rule turns per-frame detections into Events. Optional flush() emits any
final aggregate events (e.g. a final count) when the stream ends.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from core.types import Detection, Event, FrameContext


class Rule(Protocol):
    def process(self, ctx: FrameContext, dets: list[Detection]) -> Iterable[Event]:
        ...

    def flush(self) -> Iterable[Event]:
        ...
