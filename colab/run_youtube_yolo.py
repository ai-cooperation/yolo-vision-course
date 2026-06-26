"""Self-contained Colab runner: config.json -> YOLO -> annotated video + report.json.

Reads WORK/config.json (set the YouTube URL there), downloads the clip, runs
YOLO + ByteTrack, writes an annotated video and a recognition report back to
Drive. Designed to be run on a Colab GPU runtime:

    cd /content/drive/MyDrive/yolo-course
    pip -q install ultralytics yt-dlp
    python run_youtube_yolo.py

config.json keys:
    youtube_url : public YouTube URL (used unless video_path is set)
    video_path  : optional local/Drive video file (skips download; for testing)
    max_sec     : seconds to keep from the start (default 20)
    conf        : confidence threshold (default 0.35)
    classes     : null (all) or list of COCO ids, e.g. [0,2,5,7]
    model       : weights (default yolo11n.pt)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys


def _work_dir() -> str:
    if os.environ.get("YOLO_WORK"):
        return os.environ["YOLO_WORK"]
    if os.path.isdir("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/yolo-course"
    return os.path.dirname(os.path.abspath(__file__))


def _ensure(pkg: str, import_name: str | None = None) -> None:
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)


def _download(url: str, max_sec: int, vids: str) -> str:
    clip = os.path.join(vids, "clip.mp4")
    # try section download (needs ffmpeg, present on Colab); fall back to full
    section = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "mp4[height<=720]",
         "--download-sections", f"*0-{max_sec}", "--force-keyframes-at-cuts",
         "-o", clip, url]
    )
    if section.returncode != 0 or not os.path.exists(clip):
        full = os.path.join(vids, "source.mp4")
        subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "mp4[height<=720]",
                        "-o", full, url], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", full,
                        "-t", str(max_sec), "-c", "copy", clip], check=True)
    return clip


def main() -> None:
    work = _work_dir()
    cfg = json.load(open(os.path.join(work, "config.json"), encoding="utf-8"))
    vids = os.path.join(work, "videos")
    out = os.path.join(work, "outputs")
    os.makedirs(vids, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    _ensure("ultralytics")
    max_sec = int(cfg.get("max_sec", 20))
    conf = float(cfg.get("conf", 0.35))
    classes = cfg.get("classes")
    model_name = cfg.get("model", "yolo11n.pt")

    video = cfg.get("video_path")
    if not video:
        _ensure("yt-dlp", "yt_dlp")
        video = _download(cfg["youtube_url"], max_sec, vids)
    print("source video:", video)

    from collections import Counter

    from ultralytics import YOLO

    model = YOLO(model_name)
    counts: Counter = Counter()
    tracks: dict[str, set] = {}
    for r in model.track(video, tracker="bytetrack.yaml", conf=conf, classes=classes,
                         stream=True, save=True, project=out, name="run", exist_ok=True):
        for b in r.boxes:
            nm = model.names[int(b.cls)]
            counts[nm] += 1
            if b.id is not None:
                tracks.setdefault(nm, set()).add(int(b.id))

    annotated = sorted(glob.glob(f"{out}/run/*.mp4") + glob.glob(f"{out}/run/*.avi"))
    report = {
        "source": video,
        "model": model_name,
        "detections_per_class": dict(counts.most_common()),
        "unique_tracks_per_class": {k: len(v) for k, v in tracks.items()},
        "annotated_video": annotated[-1] if annotated else None,
    }
    report_path = os.path.join(work, "report.json")
    json.dump(report, open(report_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n=== recognition report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nreport written:", report_path)


if __name__ == "__main__":
    main()
