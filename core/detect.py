"""Detect layer (shared, swappable model): YOLO + built-in ByteTrack.

This is pluggable slot #1. The *layer* is identical across profiles; only the
weights change:
  - home_sensing   -> COCO pretrained (yolo11n.pt), no training needed
  - factory_count  -> self-trained best.pt (Copy-Paste synthetic + fine-tune)

Tracking uses ultralytics' built-in ByteTrack (tracker="bytetrack.yaml") so the
same code maintains stable ids for both "dedupe alerts" and "count once".
"""
from __future__ import annotations

from ultralytics import YOLO

from .types import Detection


class Detector:
    def __init__(
        self,
        model: str = "yolo11n.pt",
        conf: float = 0.35,
        classes=None,
        tracking: bool = True,
        tracker: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model)
        self.conf = conf
        self.tracking = tracking
        self.tracker = tracker
        self.classes = self._resolve_classes(classes)

    def _resolve_classes(self, classes):
        """Accept class names or ints; return list[int] or None (= all)."""
        if classes is None:
            return None
        name_to_id = {v: k for k, v in self.model.names.items()}
        out = []
        for c in classes:
            if isinstance(c, int):
                out.append(c)
            elif c in name_to_id:
                out.append(name_to_id[c])
            else:
                raise ValueError(f"unknown class {c!r}; not in model.names")
        return out

    def detect(self, frame, offset=(0, 0)) -> list[Detection]:
        ox, oy = offset
        if self.tracking:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                classes=self.classes,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame, conf=self.conf, classes=self.classes, verbose=False
            )

        out: list[Detection] = []
        boxes = results[0].boxes
        for b in boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            tid = int(b.id[0]) if (self.tracking and b.id is not None) else None
            out.append(
                Detection(
                    cls_name=self.model.names[int(b.cls)],
                    conf=float(b.conf),
                    xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                    track_id=tid,
                )
            )
        return out
