# YOLO 視覺感測 — 智慧製造課程進階模組（實作 + 驗證）

一套標準 pipeline，三個應用階梯。差異只在**標的物**與**訓練資料**，pipeline 本體共用。
完整設計見 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 三階梯

| 階梯 | 標的物 | 來源 | 訓練 | 產出 |
|---|---|---|---|---|
| 1 教學 | 標準（COCO） | YouTube 公開影片 | 否 | 辨識驗證 + 訓練串接 |
| 2 居家 | 通用（人/車/寵物） | ipcam RTSP | 否 | 事件分流 + TG |
| 3 工業 | 專用（自家產品） | 工廠 RTSP | 是（合成+fine-tune） | 穿線計數 → MES |

## 從哪開始（雲端優先，先做教學階梯）

開 [notebooks/tier1_teaching_youtube_yolo.ipynb](notebooks/tier1_teaching_youtube_yolo.ipynb) 到 Colab：
1. 執行階段選 T4 GPU
2. 掛 Google Drive（影片/模型全放 Drive，不佔本地）
3. 設一支 YouTube 公開影片 → 跑 Part A 辨識驗證
4. 要偵測 COCO 沒有的標的 → 跑 Part B 訓練串接

## 本地 / Edge 跑法（同一套 core 程式）

```bash
uv venv --python 3.12 .venv          # 本地開發 / RPi 邊緣才需要
uv pip install --python .venv/bin/python ultralytics pyyaml
.venv/bin/python run.py --profile profiles/teaching_standard.yaml --source clip.mp4 --max-frames 300
.venv/bin/python run.py --profile profiles/home_sensing.yaml     --source rtsp://...
.venv/bin/python run.py --profile profiles/factory_counting.yaml --source clip.mp4
```

## 結構

```
core/        共用四層：ingest / gate / detect+track / pipeline + profile loader
rules/       插槽2 可插拔：event_router(居家) / line_crossing(工業)
sinks/       插槽3 可插拔：console / csv / telegram (mqtt/http 之後補)
profiles/    三階梯各一份 yaml（描述三個插槽怎麼配）
notebooks/   Colab 教學實作（先跑這個）
run.py       本地/Colab CLI runner
```

切換階梯 = 換 `profiles/*.yaml`，core 不動。
