"""Gate layer (shared): cheap pre-filter before the expensive detector.

Two jobs:
  1. ROI crop  -> only look where products / subjects appear.
  2. Motion gate (MOG2 background subtraction) -> don't wake YOLO on a still
     frame; also rejects "leaves swaying" style ambient noise.

Returns (should_process, crop, offset). offset maps crop coords back to the
full frame so detections stay in full-frame pixel space.
"""
from __future__ import annotations

import cv2


def _roi_to_px(roi, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    # fractions (<=1.0) are scaled to pixels; otherwise treated as pixels
    if max(roi) <= 1.0:
        return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    return int(x1), int(y1), int(x2), int(y2)


class MotionGate:
    def __init__(self, motion: bool = True, min_area: int = 500, roi=None):
        self.motion = motion
        self.min_area = min_area
        self.roi = roi
        self._bg = (
            cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            if motion
            else None
        )

    def apply(self, frame) -> tuple[bool, "cv2.typing.MatLike", tuple[int, int]]:
        h, w = frame.shape[:2]
        if self.roi is not None:
            x1, y1, x2, y2 = _roi_to_px(self.roi, w, h)
            crop = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            crop, offset = frame, (0, 0)

        if not self.motion:
            return True, crop, offset

        mask = self._bg.apply(crop)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
        moved = int(cv2.countNonZero(mask)) >= self.min_area
        return moved, crop, offset
