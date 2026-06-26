"""Console sink: print events as they happen."""
from __future__ import annotations

from core.types import Event


class ConsoleSink:
    def emit(self, event: Event) -> None:
        extra = " ".join(f"{k}={v}" for k, v in event.data.items())
        print(f"[{event.kind:5}] frame={event.frame_index:>5} {event.label:<12} {extra}")

    def close(self) -> None:
        pass
