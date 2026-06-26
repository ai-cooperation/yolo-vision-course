"""Ingest layer (shared): turn any source into a stream of frames.

Source is one of:
  - "0" / 1 / ...        -> local webcam index
  - path/to/video.mp4    -> video file
  - rtsp://...           -> RTSP camera (factory + home both land here later)
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import cv2

from .types import FrameContext


def _resolve(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    if source.isdigit():
        return int(source)
    return source


class FrameSource:
    """Iterable yielding (FrameContext, bgr_frame) tuples."""

    def __init__(self, source: str | int, max_frames: int | None = None):
        self.source = source
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[tuple[FrameContext, "cv2.typing.MatLike"]]:
        cap = cv2.VideoCapture(_resolve(self.source))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source: {self.source!r}")
        try:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                ctx = FrameContext(
                    index=index, timestamp=time.time(), width=w, height=h
                )
                yield ctx, frame
                index += 1
                if self.max_frames is not None and index >= self.max_frames:
                    break
        finally:
            cap.release()
