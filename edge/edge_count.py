#!/usr/bin/env python3
"""
邊緣部署示範：在「你自己的電腦」上即時跑 YOLO（免 Raspberry Pi）

Raspberry Pi 只是其中一種邊緣裝置；你的筆電/桌機就是現成的邊緣裝置，而且更快。
同一套偵測 + 追蹤 + 過線計數，可跑在：網路攝影機、政府國道即時串流、或本機影片檔。

安裝：
    pip install ultralytics opencv-python lap

用法：
    python edge_count.py                    # 預設：國道1號 即時影像
    python edge_count.py --source 0         # 用你的網路攝影機（對著馬路/桌面都行）
    python edge_count.py --source clip.mp4  # 用本機影片檔
    python edge_count.py --line 0.6         # 計數線在畫面高度 60% 處
    python edge_count.py --frames 5         # 無螢幕環境：處理 5 幀後存檔結束（測試用）

視窗開啟後按 q 或 ESC 結束。
"""
import argparse
import os
import time

import cv2
import numpy as np

FREEWAY = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=10001"


def grab_mjpeg(url, timeout=8):
    """國道 MJPEG：每次抓一張最新的 JPEG（比 cv2 直連穩）。失敗回 None。"""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        # 只讀前 300KB：足夠涵蓋串流中的第一張 JPEG，避免讀 multipart 串流卡到 EOF
        data = urllib.request.urlopen(req, timeout=timeout).read(300000)
    except Exception:
        return None
    start = data.find(b"\xff\xd8")
    end = data.find(b"\xff\xd9", start)
    if start >= 0 and end >= 0:
        return cv2.imdecode(np.frombuffer(data[start:end + 2], np.uint8), cv2.IMREAD_COLOR)
    return None


def resolve_youtube(url):
    """本機（住宅 IP）上 YouTube 可用：用 yt-dlp 解析直接串流網址。"""
    import subprocess
    import sys
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])
    out = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "-f", "best[ext=mp4]/best", "-g", url],
        capture_output=True, text=True, timeout=90,
    )
    return out.stdout.strip().split("\n")[0]


def parse_args():
    ap = argparse.ArgumentParser(description="本機邊緣部署：YOLO 即時偵測 + 過線計數")
    ap.add_argument("--source", default=FREEWAY,
                    help="0=網路攝影機 / 影片檔 / 串流網址（預設：國道即時）")
    ap.add_argument("--model", default="yolo11n.pt", help="權重（預設 yolo11n.pt，會自動下載）")
    ap.add_argument("--line", type=float, default=0.5, help="計數線 y 位置，畫面高度比例 0~1")
    ap.add_argument("--conf", type=float, default=0.35, help="信心度門檻")
    ap.add_argument("--frames", type=int, default=0,
                    help=">0：處理 N 幀後存檔結束（無螢幕/測試用，不開視窗）")
    ap.add_argument("--outdir", default="edge_out", help="--frames 模式的輸出資料夾")
    return ap.parse_args()


def open_capture(source):
    """非 MJPEG 來源：開啟 cv2 擷取器。回 (cap, is_mjpeg)。"""
    if "abs2mjpg" in source:
        return None, True
    if "youtube.com" in source or "youtu.be" in source:
        source = resolve_youtube(source)
    target = int(source) if source.isdigit() else source
    if isinstance(target, int) and os.name == "nt":   # Windows 攝影機用 DirectShow 較穩
        return cv2.VideoCapture(target, cv2.CAP_DSHOW), False
    return cv2.VideoCapture(target), False


def next_frame(cap, source, is_mjpeg):
    """取下一幀。MJPEG 走 HTTP 抓圖；其餘走 cv2。回 frame 或 None（結束/失敗）。"""
    if is_mjpeg:
        return grab_mjpeg(source)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)

    cap, is_mjpeg = open_capture(args.source)
    headless = args.frames > 0
    if headless:
        os.makedirs(args.outdir, exist_ok=True)

    last_cy = {}      # 每個 track id 的上一幀中心 y
    crossed = set()   # 已計數過的 id（避免重複計）
    count = 0
    saved = 0
    misses = 0

    print("啟動邊緣偵測… 來源:", args.source)
    while True:
        frame = next_frame(cap, args.source, is_mjpeg)
        if frame is None:
            misses += 1
            if not is_mjpeg or misses > 5:   # 影片播完 / 串流連續失敗 → 結束
                break
            time.sleep(1)
            continue
        misses = 0

        h, w = frame.shape[:2]
        line_y = int(args.line * h)
        res = model.track(frame, persist=True, conf=args.conf,
                          tracker="bytetrack.yaml", verbose=False)[0]
        annotated = res.plot()
        cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 2)

        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().tolist()
            centers = res.boxes.xywh.cpu().numpy()
            for tid, (cx, cy, bw, bh) in zip(ids, centers):
                prev = last_cy.get(tid)
                if prev is not None and tid not in crossed and (prev < line_y <= cy or cy < line_y <= prev):
                    crossed.add(tid)
                    count += 1
                last_cy[tid] = cy

        cv2.putText(annotated, f"Crossed: {count}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        if headless:
            cv2.imwrite(os.path.join(args.outdir, f"frame_{saved:03d}.jpg"), annotated)
            saved += 1
            if saved >= args.frames:
                print(f"已存 {saved} 幀到 {args.outdir}/，過線計數 = {count}")
                break
        else:
            cv2.imshow("YOLO Edge - q 結束", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    if cap is not None:
        cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print("結束。總過線計數 =", count)


if __name__ == "__main__":
    main()
