"""Profile loader: build a Pipeline from a YAML profile.

A profile is the only thing that differs between application domains. It names
the detector weights, the rule list (slot #2), and the sink list (slot #3).
The core layers (ingest/gate/detect/track/pipeline) are shared verbatim.
"""
from __future__ import annotations

import yaml

from rules.detection_summary import DetectionSummary
from rules.event_router import EventRouter
from rules.line_crossing import LineCrossingCounter
from sinks.console import ConsoleSink
from sinks.csv_sink import CsvSink
from sinks.telegram import TelegramSink

from .detect import Detector
from .gate import MotionGate
from .ingest import FrameSource
from .pipeline import Pipeline

RULES = {
    "detection_summary": DetectionSummary,
    "event_router": EventRouter,
    "line_crossing": LineCrossingCounter,
}
SINKS = {"console": ConsoleSink, "csv": CsvSink, "telegram": TelegramSink}


def _build(registry, spec):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in registry:
        raise ValueError(f"unknown component {kind!r}; known: {list(registry)}")
    return registry[kind](**spec)


def load_pipeline(path: str, source=None, max_frames=None) -> Pipeline:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    src = source or cfg.get("source")
    if src is None:
        raise ValueError("no source: set it in the profile or pass --source")

    detector = Detector(**(cfg.get("detector") or {}))
    gate = MotionGate(**(cfg.get("gate") or {}))
    rules = [_build(RULES, r) for r in cfg.get("rules", [])]
    sinks = [_build(SINKS, s) for s in cfg.get("sinks", [])]

    return Pipeline(
        source=FrameSource(src, max_frames=max_frames),
        gate=gate,
        detector=detector,
        rules=rules,
        sinks=sinks,
    )
