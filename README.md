# YOLO 物件偵測完整課程

從科普、理論、架構演進，到實務操作的 YOLO 物件偵測公開教學。用公開影片 + Google Colab（免費 GPU）+ Google Drive，從零打通電腦視覺。

**線上課程網站**：http://cooperation.tw/yolo-vision-course/

## 課程章節（`docs/`）

1. 科普入門 — 電腦怎麼「看」、能做什麼
2. 核心理論 — 四種任務（分類/偵測/分割/姿態）、標籤解剖、IoU/NMS、資料標註、追蹤
3. 架構與演進 — YOLO 原理、v1→v11 時間軸、模型大小取捨
4. 場景示範 — 多類別/移動遮擋追蹤/分割，標籤前後對照（Colab 實際產出）
5. 實務操作 — 標準/通用/工業三級別、循序步驟、Colab 實作與訓練、邊緣部署
6. 來源與授權

## 動手做（Colab Notebooks，`notebooks/`）

| Notebook | 內容 |
|---|---|
| `tier1_teaching_youtube_yolo.ipynb` | 入門：YouTube 影片 → 偵測 → 追蹤 → 辨識報告 |
| `workshop/00_setup.ipynb` … `03_challenge.ipynb` | 環境/串流、偵測、應用（人數/OCR/車流）、挑戰賽 |
| `make_demos_colab.ipynb` | 對 CC0 影片標籤產生 before/after（本站第 4 章示範來源）|

## 可重用 pipeline（`core/` `rules/` `sinks/` `profiles/`）

一條標準 pipeline、三個 profile（教學/居家/工業），切換階梯只換 `profiles/*.yaml`：

```bash
python run.py --profile profiles/teaching_standard.yaml --source clip.mp4
python run.py --profile profiles/home_sensing.yaml      --source rtsp://...
python run.py --profile profiles/factory_counting.yaml  --source clip.mp4
```

## 授權

程式碼與教材供教學使用。示範影片素材與模型工具的授權見[來源頁](http://cooperation.tw/yolo-vision-course/06-credits.html)（含 OpenCV Apache-2.0、Wikimedia CC BY-SA / 公有領域、Ultralytics YOLO11 AGPL-3.0）。
