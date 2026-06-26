"""Turn a CC0/CC-BY source clip into a web-ready before/after demo.

  python make_demo.py <src> <name> <title>

Pipeline: ffmpeg trim+scale -> YOLO track (stream) -> side-by-side from
r.orig_img + r.plot() (CJK header) -> H264 faststart mp4 + poster jpg.
Outputs to site_assets/demos/.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, NAME, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
SEC, H, FPS, CONF = 15, 360, 15, 0.30
OUT = Path("site_assets/demos"); OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("site_assets/_tmp"); TMP.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# 1) trim + scale -> before.mp4 (H264)
before = TMP / f"{NAME}_before.mp4"
run(["ffmpeg", "-y", "-i", SRC, "-t", str(SEC), "-an", "-r", str(FPS),
     "-vf", f"scale=-2:{H}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(before)])

# 2) YOLO track (stream) -> build side-by-side from orig + plot
from ultralytics import YOLO  # noqa: E402

model = YOLO("yolo11n.pt")
HEAD = 44
header = None
vw = None
poster = None
frames = []
for r in model.track(str(before), tracker="bytetrack.yaml", conf=CONF, stream=True, verbose=False):
    orig = r.orig_img
    ann = r.plot()  # BGR with boxes + class + conf + track id
    Hh, W = orig.shape[:2]
    if header is None:
        hdr = Image.new("RGB", (W * 2, HEAD), (20, 20, 20)); d = ImageDraw.Draw(hdr)
        d.rectangle([0, 0, W, HEAD], fill=(70, 70, 70)); d.rectangle([W, 0, W * 2, HEAD], fill=(20, 115, 65))
        def ctext(cx, t):
            bb = d.textbbox((0, 0), t, font=FONT)
            d.text((cx - (bb[2]-bb[0]) / 2, (HEAD-(bb[3]-bb[1]))/2-bb[1]), t, font=FONT, fill=(255, 255, 255))
        ctext(W / 2, "標籤前｜原始"); ctext(W + W / 2, "標籤後｜YOLO")
        header = cv2.cvtColor(np.array(hdr), cv2.COLOR_RGB2BGR)
        vw = cv2.VideoWriter(str(TMP / f"{NAME}_sbs.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, Hh + HEAD))
    body = np.hstack([orig, ann]); cv2.line(body, (W, 0), (W, Hh), (255, 255, 255), 2)
    fr = np.vstack([header, body]); vw.write(fr); frames.append(fr)
vw.release()
poster = frames[len(frames) // 2] if frames else None

# 3) transcode to web H264 + poster
final = OUT / f"{NAME}.mp4"
run(["ffmpeg", "-y", "-i", str(TMP / f"{NAME}_sbs.mp4"), "-c:v", "libx264", "-crf", "26",
     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
if poster is not None:
    cv2.imwrite(str(OUT / f"{NAME}.jpg"), poster, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"OK {NAME}: {final} ({final.stat().st_size//1024} KB), frames={len(frames)}, title={TITLE!r}")
