"""Generate all teaching illustration images for the course site.

Outputs to docs/media/:
  label_anatomy.png      (fixed: chip clamped inside the image panel)
  human_vs_computer.png  (whole image vs pixel grid of numbers)
  annotation.png         (bbox vs polygon/mask)
  detect_run.png         (grid / IoU / NMS, 3 panels)
  app_traffic.png, app_life.png  (annotated application stills)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def F(px): return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", px)

OUT = Path("docs/media"); OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("site_assets/_tmp"); TMP.mkdir(parents=True, exist_ok=True)
BG = (14, 17, 22); MUT = (154, 167, 180); WHT = (230, 237, 243); GREEN = (31, 174, 90)
ORANGE = (255, 150, 40); BLUE = (60, 150, 255)

sample = str(TMP / "sample.jpg")
if not Path(sample).exists():
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", sample)
cap = cv2.VideoCapture("site_assets/raw/pedestrians.avi"); cap.set(1, 70); _, PED = cap.read(); cap.release()
det = YOLO("yolo11n.pt"); seg = YOLO("yolo11n-seg.pt")


def cv2pil(im): return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def person_crop(frame, pad_top=80, pad=55):
    r = det.predict(frame, classes=[0], conf=0.4, verbose=False)[0]
    b = sorted(r.boxes, key=lambda x: float(x.conf), reverse=True)[0]
    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0]); H, W = frame.shape[:2]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad_top)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
    return frame[cy1:cy2, cx1:cx2], (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)


# ---------- 1. label anatomy (chip clamped) ----------
def anatomy():
    crop, (bx1, by1, bx2, by2) = person_crop(PED)
    S = 440 / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * S), 440), interpolation=cv2.INTER_CUBIC)
    bx1, by1, bx2, by2 = bx1 * S, by1 * S, bx2 * S, by2 * S
    left = cv2pil(crop); lw, lh = left.size
    LEGW = 430; PAD = 24
    canvas = Image.new("RGB", (lw + LEGW + PAD * 2, lh + PAD * 2), BG); d = ImageDraw.Draw(canvas)
    canvas.paste(left, (PAD, PAD)); ox = oy = PAD
    d.rectangle([ox + bx1, oy + by1, ox + bx2, oy + by2], outline=GREEN, width=4)
    parts = [("id:3", BLUE), ("person", WHT), ("0.80", ORANGE)]
    fch = F(28); pad = 12; gap = 16
    ws = [d.textbbox((0, 0), t, font=fch)[2] for t, _ in parts]
    chw = sum(ws) + gap * 2 + pad * 2; chh = 46
    chx = ox + bx1
    chx = min(chx, ox + lw - chw - 4); chx = max(ox + 2, chx)   # clamp inside image
    chy = oy + max(2, by1 - chh - 8)
    d.rectangle([chx, chy, chx + chw, chy + chh], fill=(18, 22, 28))
    cur = chx + pad
    for (t, col), w in zip(parts, ws):
        d.text((cur, chy + 8), t, font=fch, fill=col); cur += w + gap
    lx = lw + PAD * 2; ly = PAD + 24
    d.text((lx, PAD - 6), "一個偵測標籤的四個部分", font=F(24), fill=MUT)
    rows = [(GREEN, "框 Bounding Box", "物體在哪裡（位置與大小）"),
            (WHT, "類別 Class", "這是什麼：person / car / dog…"),
            (ORANGE, "信心度 Confidence", "0~1，越接近 1 模型越確定"),
            (BLUE, "追蹤 ID", "跨影格認出同一個物體（ByteTrack）")]
    for col, term, desc in rows:
        d.ellipse([lx, ly + 4, lx + 22, ly + 26], fill=col)
        d.text((lx + 36, ly), term, font=F(23), fill=WHT)
        d.text((lx + 36, ly + 34), desc, font=F(19), fill=MUT); ly += 92
    canvas.save(OUT / "label_anatomy.png"); print("anatomy", canvas.size)


# ---------- 2. human vs computer (pixel grid) ----------
def human_vs_computer():
    crop, (bx1, by1, bx2, by2) = person_crop(PED, pad_top=10, pad=10)
    Lh = 380; S = Lh / crop.shape[0]
    leftim = cv2.resize(crop, (int(crop.shape[1] * S), Lh))
    # small pixel block from torso
    py, px = (by1 + by2) // 2, (bx1 + bx2) // 2
    N = 8; block = crop[py:py + N, px:px + N].copy()
    if block.shape[0] < N or block.shape[1] < N:
        block = crop[:N, :N].copy()
    cell = 60; gw = N * cell
    grid = Image.new("RGB", (gw, gw), BG); dg = ImageDraw.Draw(grid)
    fpx = F(13)
    for yy in range(N):
        for xx in range(N):
            b, g, r = block[yy, xx]
            dg.rectangle([xx * cell, yy * cell, (xx + 1) * cell, (yy + 1) * cell], fill=(int(r), int(g), int(b)), outline=(40, 46, 54))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            tc = (20, 20, 20) if lum > 130 else (235, 235, 235)
            dg.text((xx * cell + 4, yy * cell + 8), f"{int(r)}\n{int(g)}\n{int(b)}", font=fpx, fill=tc)
    HEAD = 50; GAP = 16
    left = cv2pil(leftim); lw, lh = left.size
    W = lw + GAP + gw + 48; H = max(lh, gw) + HEAD + 48
    canvas = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(canvas)
    d.text((24, 16), "人看：整張圖", font=F(22), fill=GREEN)
    d.text((24 + lw + GAP, 16), "電腦看：每個像素就是 RGB 數字", font=F(22), fill=ORANGE)
    canvas.paste(left, (24, HEAD)); canvas.paste(grid, (24 + lw + GAP, HEAD))
    canvas.save(OUT / "human_vs_computer.png"); print("human_vs_computer", canvas.size)


# ---------- 3. annotation: bbox vs polygon ----------
def annotation():
    crop, (bx1, by1, bx2, by2) = person_crop(PED, pad_top=20, pad=25)
    S = 420 / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * S), 420))
    a = crop.copy(); cv2.rectangle(a, (int(bx1 * S), int(by1 * S)), (int(bx2 * S), int(by2 * S)), (90, 174, 31), 3)
    # polygon from seg
    rs = seg.predict(crop, classes=[0], conf=0.35, verbose=False)[0]
    bpoly = crop.copy()
    if rs.masks is not None and len(rs.masks.xy):
        poly = max(rs.masks.xy, key=lambda p: len(p)).astype(np.int32)
        cv2.polylines(bpoly, [poly], True, (255, 150, 60), 2)
        overlay = bpoly.copy(); cv2.fillPoly(overlay, [poly], (255, 150, 60)); bpoly = cv2.addWeighted(overlay, 0.25, bpoly, 0.75, 0)
    HEAD = 50; GAP = 12; W = crop.shape[1]; H = crop.shape[0]
    canvas = Image.new("RGB", (W * 2 + GAP + 48, H + HEAD + 48), BG); d = ImageDraw.Draw(canvas)
    d.text((24, 16), "方框 bbox：框住物體", font=F(22), fill=GREEN)
    d.text((24 + W + GAP, 16), "多邊形／遮罩：描出輪廓", font=F(22), fill=BLUE)
    canvas.paste(cv2pil(a), (24, HEAD)); canvas.paste(cv2pil(bpoly), (24 + W + GAP, HEAD))
    d.text((24, HEAD + H + 10), "偵測用方框，快又便宜；分割用多邊形/遮罩，得到像素級邊緣。", font=F(18), fill=MUT)
    canvas.save(OUT / "annotation.png"); print("annotation", canvas.size)


# ---------- 4. detect_run: grid / IoU / NMS ----------
def detect_run():
    base, (bx1, by1, bx2, by2) = person_crop(PED, pad_top=30, pad=30)
    S = 300 / base.shape[0]; base = cv2.resize(base, (int(base.shape[1] * S), 300))
    bx1, by1, bx2, by2 = [int(v * S) for v in (bx1, by1, bx2, by2)]
    Hh, Ww = base.shape[:2]
    # grid panel
    g = base.copy()
    for x in range(0, Ww, Ww // 6): cv2.line(g, (x, 0), (x, Hh), (255, 255, 255), 1)
    for y in range(0, Hh, Hh // 8): cv2.line(g, (0, y), (Ww, y), (255, 255, 255), 1)
    # iou panel (synthetic)
    iou = np.full((Hh, Ww, 3), 22, np.uint8)
    r1 = (Ww // 6, Hh // 4, Ww // 6 + Ww // 2, Hh // 4 + Hh // 2)
    r2 = (Ww // 3, Hh // 3, Ww // 3 + Ww // 2, Hh // 3 + Hh // 2)
    ix1, iy1 = max(r1[0], r2[0]), max(r1[1], r2[1]); ix2, iy2 = min(r1[2], r2[2]), min(r1[3], r2[3])
    cv2.rectangle(iou, (ix1, iy1), (ix2, iy2), (60, 90, 120), -1)
    cv2.rectangle(iou, r1[:2], r1[2:], (255, 150, 40), 2); cv2.rectangle(iou, r2[:2], r2[2:], (60, 150, 255), 2)
    # nms panel: many boxes -> implied one (draw offsets)
    nms = base.copy()
    for dx, dy in [(-8, -6), (6, -10), (-4, 8), (10, 4)]:
        cv2.rectangle(nms, (bx1 + dx, by1 + dy), (bx2 + dx, by2 + dy), (120, 120, 120), 1)
    cv2.rectangle(nms, (bx1, by1), (bx2, by2), (90, 174, 31), 3)
    panels = [(g, "掃格子 Grid", "每格猜：附近有沒有物體"),
              (iou, "IoU 重疊度", "交集 ÷ 聯集"),
              (nms, "NMS 去重", "多個框 → 留最好的一個")]
    HEAD = 52; GAP = 12
    W = Ww * 3 + GAP * 2 + 32; H = Hh + HEAD + 24
    canvas = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(canvas)
    for i, (im, t, sub) in enumerate(panels):
        x = 16 + i * (Ww + GAP)
        d.text((x, 12), t, font=F(21), fill=GREEN); d.text((x, 38), sub, font=F(15), fill=MUT)
        canvas.paste(cv2pil(im), (x, HEAD))
    canvas.save(OUT / "detect_run.png"); print("detect_run", canvas.size)


# ---------- 5. application stills ----------
def app_still(src_video, out_name, sec_frame=60):
    cap = cv2.VideoCapture(src_video); cap.set(1, sec_frame); ok, fr = cap.read(); cap.release()
    if not ok:
        cap = cv2.VideoCapture(src_video); ok, fr = cap.read(); cap.release()
    fr = cv2.resize(fr, (640, int(640 * fr.shape[0] / fr.shape[1])))
    r = det.predict(fr, conf=0.35, verbose=False)[0]
    cv2.imwrite(str(OUT / out_name), r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out_name)


anatomy(); human_vs_computer(); annotation(); detect_run()
app_still("site_assets/raw/street_lagos.webm", "app_traffic.jpg", 90)
app_still("site_assets/raw/pedestrians.avi", "app_life.jpg", 70)
print("all illustrations done")
