"""Compose a web demo from Colab-produced before.mp4 + after.mp4 (layout only).

  python compose_demo.py <name> <title>

Reads site_assets/colab_demos/<name>/{before,after}.mp4 (pulled from Drive),
stacks them side-by-side with a CJK header, writes docs/media/<name>.mp4 (H264)
+ <name>.jpg poster. This does NOT run YOLO — labelling happens on Colab.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME, TITLE = sys.argv[1], sys.argv[2]
SRC = Path("site_assets/colab_demos") / NAME
OUT = Path("docs/media"); OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("site_assets/_tmp"); TMP.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
FPS, HEAD = 15, 44


def run(c):
    subprocess.run(c, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


c1 = cv2.VideoCapture(str(SRC / "before.mp4"))
c2 = cv2.VideoCapture(str(SRC / "after.mp4"))
W = int(c1.get(3)); H = int(c1.get(4))
hdr = Image.new("RGB", (W * 2, HEAD), (20, 20, 20)); d = ImageDraw.Draw(hdr)
d.rectangle([0, 0, W, HEAD], fill=(70, 70, 70)); d.rectangle([W, 0, W * 2, HEAD], fill=(20, 115, 65))
def ctext(cx, t):
    bb = d.textbbox((0, 0), t, font=FONT)
    d.text((cx - (bb[2]-bb[0]) / 2, (HEAD-(bb[3]-bb[1]))/2-bb[1]), t, font=FONT, fill=(255, 255, 255))
ctext(W / 2, "標籤前｜原始"); ctext(W + W / 2, "標籤後｜YOLO")
header = cv2.cvtColor(np.array(hdr), cv2.COLOR_RGB2BGR)

tmp = TMP / f"{NAME}_sbs.mp4"
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H + HEAD))
n = min(int(c1.get(7)), int(c2.get(7))); poster = None
for i in range(n):
    o1, f1 = c1.read(); o2, f2 = c2.read()
    if not (o1 and o2):
        break
    if f2.shape[:2] != (H, W):
        f2 = cv2.resize(f2, (W, H))
    body = np.hstack([f1, f2]); cv2.line(body, (W, 0), (W, H), (255, 255, 255), 2)
    fr = np.vstack([header, body]); vw.write(fr)
    if i == n // 2:
        poster = fr.copy()
vw.release(); c1.release(); c2.release()

run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "26",
     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"{NAME}.mp4")])
if poster is not None:
    cv2.imwrite(str(OUT / f"{NAME}.jpg"), poster, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"OK {NAME} -> docs/media/{NAME}.mp4 (+ poster), title={TITLE!r}")
