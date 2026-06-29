#!/usr/bin/env python3
"""
用筆電視訊鏡頭玩「猜拳」——MediaPipe 手部 21 關鍵點辨識石頭 / 布 / 剪刀

為什麼用 MediaPipe 而不是 YOLO？
  YOLO 擅長「框出物件」（車、人、零件）；但手勢的差別在「手指彎不彎」，
  用手部關鍵點（landmarks）判斷更準、更穩，而且免訓練、免資料集、免授權問題。
  「依任務選對工具」正是這門課的重點之一。

這是「在自己電腦上跑」的第二個案例（第一個是 edge_count.py 看交通串流）。

安裝：
    pip install mediapipe opencv-python pillow

用法：
    python rps_play.py              # 用內建鏡頭
    python rps_play.py --camera 1   # 換另一支鏡頭

玩法：對鏡頭比出石頭 / 布 / 剪刀 → 按「空白鍵」出拳 → 電腦隨機出 → 判定勝負、累積比分。
按 q 或 ESC 結束。
"""
import argparse
import math
import os
import random
import urllib.request

import cv2
import numpy as np

MOVES = ("rock", "paper", "scissors")
ZH = {"rock": "石頭", "paper": "布", "scissors": "剪刀"}
EN = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}  # key 贏 value

FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (指尖, PIP) 食/中/無名/小指
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
TASK_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

# 常見中文字型位置（Mac / Windows / Linux）。找得到就用 PIL 畫中文，否則退英文。
_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(size=30):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


_FONT = _load_font()
_CJK = _FONT is not None


def name(move):
    """手勢顯示名：有中文字型用中文，否則英文。"""
    if move is None:
        return "（把手放進畫面）" if _CJK else "(show your hand)"
    return ZH[move] if _CJK else EN[move]


def put_lines(frame, lines):
    """畫多行文字。lines = [(text, (x, y), (B, G, R)), ...]。
    有中文字型 → 用 PIL 畫（支援中文）；否則退 cv2.putText（僅 ASCII）。"""
    if _CJK:
        from PIL import Image, ImageDraw
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for text, (x, y), (b, g, r) in lines:
            draw.text((x, y), text, font=_FONT, fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        for text, (x, y), color in lines:
            cv2.putText(frame, text, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


_FONT_BIG = _load_font(110)


def put_big_center(frame, text, color_bgr):
    """畫面正中央的大字（倒數 3/2/1、出拳）。PIL 優先，否則 cv2。"""
    h, w = frame.shape[:2]
    b, g, r = color_bgr
    if _CJK and _FONT_BIG is not None:
        from PIL import Image, ImageDraw
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=_FONT_BIG)
        draw.text(((w - (x1 - x0)) // 2 - x0, (h - (y1 - y0)) // 2 - y0 - 20),
                  text, font=_FONT_BIG, fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        scale, thick = 3.0, 6
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thick)


def classify(landmarks):
    """從 21 個 (x, y) 關鍵點判手勢。指尖離手腕 > PIP 離手腕 ×1.15 → 該指伸直（與朝向無關）。"""
    wrist = 0

    def dist(a, b):
        return math.hypot(landmarks[a][0] - landmarks[b][0],
                          landmarks[a][1] - landmarks[b][1])

    extended = [dist(tip, wrist) > dist(pip, wrist) * 1.15 for tip, pip in FINGERS]
    n = sum(extended)
    if n == 0:
        return "rock"
    if extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "scissors"
    if n >= 4:
        return "paper"
    return None


def judge(player, computer):
    if player == computer:
        return "draw"
    return "win" if BEATS[player] == computer else "lose"


def outcome_text(result):
    if not _CJK:
        return {"win": "YOU WIN", "lose": "CPU WINS", "draw": "DRAW"}[result]
    return {"win": "你贏！", "lose": "電腦贏", "draw": "平手"}[result]


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


class HandTracker:
    """抓手部 21 關鍵點：優先用 mediapipe 傳統 solutions API；
    新版 mediapipe 沒有 solutions 時，自動改用 Tasks API（會下載一個小模型）。"""

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        try:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
            self.drawer = mp.solutions.drawing_utils
            self.conn = mp.solutions.hands.HAND_CONNECTIONS
            self.mode = "solutions"
        except AttributeError:
            self._init_tasks()
        print(f"MediaPipe 手部偵測模式：{self.mode}")

    def _init_tasks(self):
        from mediapipe.tasks.python import vision, BaseOptions
        path = "hand_landmarker.task"
        if not os.path.exists(path):
            print("下載手部關鍵點模型…")
            urllib.request.urlretrieve(TASK_MODEL_URL, path)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE, num_hands=1)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.mode = "tasks"

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.mode == "solutions":
            res = self.hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            hand = res.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(frame_bgr, hand, self.conn)
            return [(p.x, p.y) for p in hand.landmark]
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)
        if not res.hand_landmarks:
            return None
        lm = [(p.x, p.y) for p in res.hand_landmarks[0]]
        draw_hand(frame_bgr, lm)
        return lm


def hand_crop(clean_frame, landmarks, pad=0.3, size=180):
    """以手部關鍵點的外接框 + 邊距，從乾淨畫面截一張手勢縮圖。"""
    h, w = clean_frame.shape[:2]
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    x0 = max(0, int((min(xs) - pad * bw) * w))
    x1 = min(w, int((max(xs) + pad * bw) * w))
    y0 = max(0, int((min(ys) - pad * bh) * h))
    y1 = min(h, int((max(ys) + pad * bh) * h))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return cv2.resize(clean_frame[y0:y1, x0:x1], (size, size))


def overlay_thumb(frame, thumb, label):
    """把縮圖貼到畫面右上角，加白框與標籤。"""
    h, w = frame.shape[:2]
    th, tw = thumb.shape[:2]
    x, y = w - tw - 14, 14
    frame[y:y + th, x:x + tw] = thumb
    cv2.rectangle(frame, (x - 2, y - 2), (x + tw + 2, y + th + 2), (255, 255, 255), 2)
    put_lines(frame, [(label, (x, y + th + 6), (255, 255, 255))])


WIN = "RPS - q to quit"
ORDER = ("scissors", "rock", "paper")   # 練習順序：剪刀 → 石頭 → 布
HOLD = 18                               # 手勢穩住這麼多幀就觸發（約 1 秒）
STEP = 6                                # 倒數每格幀數（顯示 3 → 2 → 1）


def countdown_num(stable_n):
    return min(3, max(1, (HOLD - stable_n) // STEP + 1))


def open_camera(index):
    """開攝影機；Windows 用 DirectShow 後端，開鏡頭較快也較穩。"""
    if os.name == "nt":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def main():
    ap = argparse.ArgumentParser(description="MediaPipe 視訊猜拳")
    ap.add_argument("--camera", type=int, default=0, help="鏡頭編號（預設 0）")
    args = ap.parse_args()

    tracker = HandTracker()
    if not _CJK:
        print("找不到中文字型，畫面文字改用英文顯示（功能不受影響）")
    cap = open_camera(args.camera)
    if not cap.isOpened():
        print("打不開鏡頭，請確認權限或換 --camera 編號")
        return

    refs = {}                 # move -> 練習階段截下的手勢縮圖
    mode = "practice"         # practice → game
    pidx = 0                  # 練習進度
    score = {"you": 0, "cpu": 0}
    banner = ""               # 持久顯示的提示 / 結果
    last_move = None
    stable_move, stable_n = None, 0
    cooldown = 0
    cpu_move = None           # 本回合 CPU 出的拳

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        clean = frame.copy()              # 沒畫骨架的乾淨畫面，給截圖用
        lm = tracker.process(frame)
        move = classify(lm) if lm else None
        H = frame.shape[0]

        if move and move == stable_move:
            stable_n += 1
        else:
            stable_move, stable_n = move, (1 if move else 0)
        if move:
            last_move = move
        if cooldown > 0:
            cooldown -= 1

        # ---------- 練習階段：依序記錄 剪刀 / 石頭 / 布 ----------
        if mode == "practice":
            target = ORDER[pidx]
            lines = [
                (f"練習 {pidx + 1}/3：請比出「{name(target)}」" if _CJK
                 else f"Practice {pidx + 1}/3: show {EN[target]}", (12, 12), (0, 255, 255)),
                (f"目前偵測：{name(move)}" if _CJK else f"Now: {name(move)}", (12, 50), (200, 200, 200)),
            ]
            if banner:
                lines.append((banner, (12, 88), (0, 255, 0)))
            lines.append(((f"比出指定手勢穩住即可記錄；q 結束" if _CJK
                           else "hold the asked gesture to save; q quit"), (12, H - 34), (200, 200, 200)))
            put_lines(frame, lines)
            if cooldown == 0 and move == target and 0 < stable_n < HOLD:
                put_big_center(frame, str(countdown_num(stable_n)), (0, 200, 255))
            cv2.imshow(WIN, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cooldown == 0 and move == target and stable_n >= HOLD:
                crop = hand_crop(clean, lm)
                if crop is not None:
                    refs[target] = crop
                    print(f"[練習] 已記錄 {ZH[target]}")
                    banner = f"已記錄 {ZH[target]}" if _CJK else f"Saved {EN[target]}"
                    pidx += 1
                    stable_n, cooldown = 0, 20
                    if pidx >= len(ORDER):
                        mode = "game"
                        banner = "準備好了！開始猜拳" if _CJK else "Ready! Let's play"
                        cooldown = 30
            continue

        # ---------- 遊戲階段 ----------
        lines = [
            (f"偵測：{name(move)}" if _CJK else f"Detected: {name(move)}", (12, 12), (0, 255, 255)),
            (f"YOU {score['you']} : {score['cpu']} CPU", (12, 50), (0, 0, 255)),
        ]
        if banner:
            lines.append((banner, (12, 88), (0, 255, 0)))
        lines.append(((f"穩住手勢倒數 3-2-1 出拳（空白鍵亦可）；q 結束" if _CJK
                       else "hold to count 3-2-1 (or SPACE); q quit"), (12, H - 34), (200, 200, 200)))
        put_lines(frame, lines)

        if cooldown > 0 and cpu_move is not None:           # 出拳後：顯示 CPU 的拳 + 結果
            overlay_thumb(frame, refs[cpu_move], "電腦出" if _CJK else "CPU")
            if cooldown >= 38:
                put_big_center(frame, "出拳！" if _CJK else "GO!", (0, 255, 0))
        elif cooldown == 0 and move and 0 < stable_n < HOLD:  # 穩住中：倒數
            put_big_center(frame, str(countdown_num(stable_n)), (0, 200, 255))
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        played = None
        if key == ord(" ") and (move or last_move):
            played = move or last_move
        elif cooldown == 0 and move and stable_n >= HOLD:
            played = move
        if played:
            cpu_move = random.choice(MOVES)
            result = judge(played, cpu_move)
            if result == "win":
                score["you"] += 1
            elif result == "lose":
                score["cpu"] += 1
            banner = (f"你{ZH[played]} vs 電腦{ZH[cpu_move]} → {outcome_text(result)}" if _CJK
                      else f"You {EN[played]} vs CPU {EN[cpu_move]} -> {outcome_text(result)}")
            print(f"[出拳] {banner}   比分 YOU {score['you']} : {score['cpu']} CPU")
            cooldown, stable_n = 45, 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"最終比分 你 {score['you']} : {score['cpu']} 電腦")


if __name__ == "__main__":
    main()
