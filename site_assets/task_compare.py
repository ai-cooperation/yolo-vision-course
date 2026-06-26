"""Four-task comparison figure: classification / detection / segmentation / pose.

Runs yolo11n, yolo11n-seg, yolo11n-pose on one sample image and lays out a 2x2
panel with CJK headers — the theory page's "different kinds of recognition".
Output: docs/media/task_compare.png
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def F(px): return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", px)

Path("site_assets/_tmp").mkdir(parents=True, exist_ok=True)
img_path = "site_assets/_tmp/sample.jpg"
if not Path(img_path).exists():
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", img_path)
src = cv2.imread(img_path)
src = cv2.resize(src, (520, int(520 * src.shape[0] / src.shape[1])))

def plot(model_name, **kw):
    r = YOLO(model_name).predict(src, verbose=False, conf=0.4, **kw)[0]
    return r.plot(img=src.copy())

# classification panel = original + conceptual label bar
cls = src.copy()
det = plot("yolo11n.pt")
seg = plot("yolo11n-seg.pt")
pose = plot("yolo11n-pose.pt")

panels = [
    (cls, "分類 Classification", "整張圖一個標籤：這是什麼"),
    (det, "偵測 Detection", "每個物體一個框：是什麼 + 在哪"),
    (seg, "分割 Segmentation", "畫出物體輪廓（遮罩/邊緣），到像素級"),
    (pose, "姿態 Pose", "找出關鍵點與骨架"),
]
W = src.shape[1]; H = src.shape[0]; HEAD = 56; GAP = 8
cell_w, cell_h = W, H + HEAD
canvas = Image.new("RGB", (cell_w * 2 + GAP, cell_h * 2 + GAP), (14, 17, 22))
d = ImageDraw.Draw(canvas)
ft = F(26); fs = F(17)
for i, (im, title, sub) in enumerate(panels):
    cx = (i % 2) * (cell_w + GAP); cy = (i // 2) * (cell_h + GAP)
    d.rectangle([cx, cy, cx + cell_w, cy + HEAD], fill=(22, 27, 34))
    d.text((cx + 14, cy + 6), title, font=ft, fill=(31, 174, 90))
    d.text((cx + 14, cy + 34), sub, font=fs, fill=(154, 167, 180))
    pim = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    canvas.paste(pim, (cx, cy + HEAD))

canvas.save("docs/media/task_compare.png")
print("OK -> docs/media/task_compare.png", canvas.size)
