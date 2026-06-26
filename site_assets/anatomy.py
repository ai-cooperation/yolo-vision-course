"""YOLO label anatomy diagram — left: a real detection; right: colour-keyed legend.

Each label element gets a distinct colour shared between the image and the
legend, so there is no number overlapping the text.
Output: docs/media/label_anatomy.png
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def F(px): return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", px)

BG = (14, 17, 22)
C_BOX = (31, 174, 90)     # 框 green
C_CLS = (240, 240, 240)   # 類別 white
C_CONF = (255, 150, 40)   # 信心度 orange
C_ID = (60, 150, 255)     # 追蹤ID blue

# 1) frame + clearest person (geometry only)
cap = cv2.VideoCapture("site_assets/raw/pedestrians.avi"); cap.set(1, 70)
ok, frame = cap.read(); cap.release()
det = YOLO("yolo11n.pt").predict(frame, classes=[0], conf=0.4, verbose=False)[0]
b = sorted(det.boxes, key=lambda x: float(x.conf), reverse=True)[0]
x1, y1, x2, y2 = (int(v) for v in b.xyxy[0]); H, W = frame.shape[:2]

# 2) crop with headroom for the chip
cx1, cy1 = max(0, x1 - 60), max(0, y1 - 80)
cx2, cy2 = min(W, x2 + 60), min(H, y2 + 50)
crop = frame[cy1:cy2, cx1:cx2]
target_h = 440; S = target_h / crop.shape[0]
crop = cv2.resize(crop, (int(crop.shape[1] * S), target_h), interpolation=cv2.INTER_CUBIC)
bx1, by1 = (x1 - cx1) * S, (y1 - cy1) * S
bx2, by2 = (x2 - cx1) * S, (y2 - cy1) * S

left = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGB")
lw, lh = left.size
LEGW = 430; PAD = 24
canvas = Image.new("RGB", (lw + LEGW + PAD * 2, lh + PAD * 2), BG)
canvas.paste(left, (PAD, PAD))
d = ImageDraw.Draw(canvas)
ox, oy = PAD, PAD

# 3) box
d.rectangle([ox + bx1, oy + by1, ox + bx2, oy + by2], outline=C_BOX, width=4)

# 4) chip "id:3  person  0.80"  (coloured segments)
parts = [("id:3", C_ID), ("person", C_CLS), ("0.80", C_CONF)]
fch = F(28); pad = 12; gap = 16
ws = [d.textbbox((0, 0), t, font=fch)[2] for t, _ in parts]
chw = sum(ws) + gap * 2 + pad * 2; chh = 46
chx = ox + bx1; chy = oy + max(0, by1 - chh - 8)
d.rectangle([chx, chy, chx + chw, chy + chh], fill=(18, 22, 28))
cur = chx + pad
for (t, col), w in zip(parts, ws):
    d.text((cur, chy + 8), t, font=fch, fill=col); cur += w + gap

# 5) legend (right), colour-keyed
lx = lw + PAD * 2; ly = PAD + 24
d.text((lx, PAD - 6), "一個偵測標籤的四個部分", font=F(24), fill=(154, 167, 180))
rows = [
    (C_BOX, "框 Bounding Box", "物體在哪裡（位置與大小）"),
    (C_CLS, "類別 Class", "這是什麼：person / car / dog…"),
    (C_CONF, "信心度 Confidence", "0~1，越接近 1 模型越確定"),
    (C_ID, "追蹤 ID", "跨影格認出同一個物體（ByteTrack）"),
]
ft = F(23); fd = F(19)
for col, term, desc in rows:
    d.ellipse([lx, ly + 4, lx + 22, ly + 26], fill=col)
    d.text((lx + 36, ly), term, font=ft, fill=(230, 237, 243))
    d.text((lx + 36, ly + 34), desc, font=fd, fill=(154, 167, 180))
    ly += 92

canvas.save("docs/media/label_anatomy.png")
print("OK -> docs/media/label_anatomy.png", canvas.size)
