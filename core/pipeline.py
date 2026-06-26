"""Pipeline orchestrator (shared): ingest -> gate -> detect -> rules -> sinks.

The control flow is identical for every profile. Profiles only swap the
detector weights, the rule list, and the sink list.
"""
from __future__ import annotations

from dataclasses import replace

from .detect import Detector
from .gate import MotionGate
from .ingest import FrameSource


class Pipeline:
    def __init__(self, source: FrameSource, gate: MotionGate, detector: Detector,
                 rules: list, sinks: list):
        self.source = source
        self.gate = gate
        self.detector = detector
        self.rules = rules
        self.sinks = sinks

    def run(self) -> int:
        emitted = 0
        try:
            for ctx, frame in self.source:
                should, crop, offset = self.gate.apply(frame)
                ctx = replace(ctx, motion=should)
                dets = self.detector.detect(crop, offset) if should else []
                for rule in self.rules:
                    for event in rule.process(ctx, dets):
                        for sink in self.sinks:
                            sink.emit(event)
                        emitted += 1
        finally:
            for rule in self.rules:
                for event in getattr(rule, "flush", lambda: [])():
                    for sink in self.sinks:
                        sink.emit(event)
                    emitted += 1
            for sink in self.sinks:
                sink.close()
        return emitted
