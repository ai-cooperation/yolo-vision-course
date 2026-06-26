"""Sink protocol (pluggable slot #3): where events go.

console / csv (both profiles), telegram (home), mqtt/http (factory, later).
"""
from __future__ import annotations

from typing import Protocol

from core.types import Event


class Sink(Protocol):
    def emit(self, event: Event) -> None:
        ...

    def close(self) -> None:
        ...
