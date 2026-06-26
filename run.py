"""CLI: run any profile against any source.

    python run.py --profile profiles/home_sensing.yaml --source rtsp://...
    python run.py --profile profiles/teaching_standard.yaml --source clip.mp4 --max-frames 300

Works the same locally or inside a Colab cell (the core/ library is portable).
"""
from __future__ import annotations

import argparse
import os

from core.profile import load_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO vision pipeline runner")
    ap.add_argument("--profile", required=True, help="path to a profiles/*.yaml")
    ap.add_argument("--source", default=None, help="override source (path/rtsp/webcam index)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    args = ap.parse_args()

    # ensure outputs/ exists for csv sinks
    os.makedirs("outputs", exist_ok=True)

    pipeline = load_pipeline(args.profile, source=args.source, max_frames=args.max_frames)
    emitted = pipeline.run()
    print(f"\ndone. events emitted: {emitted}")


if __name__ == "__main__":
    main()
