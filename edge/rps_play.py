#!/usr/bin/env python3
"""
用筆電視訊鏡頭玩「猜拳」——MediaPipe 手部 21 關鍵點辨識石頭 / 布 / 剪刀

為什麼用 MediaPipe 而不是 YOLO？
  YOLO 擅長「框出物件」（車、人、零件）；但手勢的差別在「手指彎不彎」，
  用手部關鍵點（landmarks）判斷更準、更穩，而且免訓練、免資料集、免授權問題。
  「依任務選對工具」正是這門課的重點之一。

這是「在自己電腦上跑」的第二個案例（第一個是 edge_count.py 看交通串流）。

安裝：
    pip install mediapipe opencv-python

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

MOVES = ("rock", "paper", "scissors")
ZH = {"rock": "石頭", "paper": "布", "scissors": "剪刀"}
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}  # key 贏 value

FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (指尖, PIP) 食/中/無名/小指
# 手部 21 點的骨架連線（畫圖用）
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),            # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),       # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),     # 無名指
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),  # 小指 + 手掌
]
TASK_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")


def classify(landmarks):
    """從 21 個 (x, y) 關鍵點判手勢。

    判「手指伸直」：指尖離手腕 > PIP 關節離手腕 → 伸直。用距離比較，與手的朝向無關。
    """
    wrist = 0  # landmark index of the wrist

    def dist(a, b):
        return math.hypot(landmarks[a][0] - landmarks[b][0],
                          landmarks[a][1] - landmarks[b][1])

    # 乘 1.15 margin：握拳時指尖只稍微比 PIP 遠也算「彎」，石頭judge更準
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
        return "平手"
    return "你贏！" if BEATS[player] == computer else "電腦贏"


def draw_hand(frame, landmarks):
    """手動把 21 點 + 骨架畫到畫面上（Tasks API 沒有現成繪圖工具時用）。"""
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
        """回 21 個 (x, y) 關鍵點（normalized）或 None，並在畫面上畫手。"""
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


def main():
    ap = argparse.ArgumentParser(description="MediaPipe 視訊猜拳")
    ap.add_argument("--camera", type=int, default=0, help="鏡頭編號（預設 0）")
    args = ap.parse_args()

    tracker = HandTracker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("打不開鏡頭，請確認權限或換 --camera 編號")
        return

    score = {"you": 0, "cpu": 0}
    banner = "對鏡頭比出手勢，按空白鍵出拳（q 結束）"
    flash = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # 鏡像，像照鏡子
        lm = tracker.process(frame)
        move = classify(lm) if lm else None

        cv2.putText(frame, f"偵測：{ZH.get(move, '（把手放進畫面）')}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"YOU {score['you']} : {score['cpu']} CPU", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if flash > 0:
            cv2.putText(frame, banner, (12, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            flash -= 1

        cv2.imshow("MediaPipe 猜拳 - 空白鍵出拳, q 結束", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if move is None:
                banner = "沒看清楚手勢，再試一次"
            else:
                cpu = random.choice(MOVES)
                outcome = judge(move, cpu)
                if outcome == "你贏！":
                    score["you"] += 1
                elif outcome == "電腦贏":
                    score["cpu"] += 1
                banner = f"你出{ZH[move]} vs 電腦{ZH[cpu]} → {outcome}"
            flash = 45

    cap.release()
    cv2.destroyAllWindows()
    print(f"最終比分 你 {score['you']} : {score['cpu']} 電腦")


if __name__ == "__main__":
    main()
