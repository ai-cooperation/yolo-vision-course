# YOLO 視覺感測基礎架構（雲端優先）

> 一套標準 pipeline，三個應用階梯。差異只在「標的物」與「訓練資料」，pipeline 本體共用。
> 雲端優先：運算用 Colab T4（免費 GPU），資料（影片/圖片/資料集/模型）放 Google Drive，
> 盡量不佔本地磁碟。本地只在最後「邊緣部署」階段才出現。

---

## 1. 三階梯：標的物決定一切

同一條 pipeline，標的物在不在 COCO 預訓練類別裡，決定要不要自己訓練。

| 階梯 | 標的物 | 影像來源 | 要訓練嗎 | 主要產出 | 對應 SB 計畫 |
|---|---|---|---|---|---|
| 1 教學 | **標準標的**（人/車/動物… COCO 內建） | YouTube 公開影片 | 否（直接用 COCO） | 辨識驗證 demo + 偵測報告 | 新建（沿用課程教學 pattern） |
| 2 居家 | **通用標的**（人/車/寵物） | 自家 ipcam RTSP | 否（COCO 起步） | 事件分流 + TG 通知 | 2026-06-04 居家 IoT 視覺感測 |
| 3 工業 | **專用標的**（自家產品，非 COCO） | 工廠 RTSP | 是（合成資料 + fine-tune） | 穿線計數 → MES/ERP | 2026-06-23 工廠 EdgeAI 計數 PoC |

教學階梯先打通「影片進 → 偵測出 → 驗證辨識」的最小閉環；居家換資料來源 + 加規則；工業換成自訓練模型 + 計數 + 邊緣部署。**學員先學會在標準標的上跑通，再換到自己的通用/專用標的——這就是課程「漸進複雜度」原則的視覺版。**

教學階梯內部還有半階「訓練串接」：當學員想偵測 COCO 沒有的標的（例如某零件），就走 Roboflow 標註 → Colab T4 訓練 → 新影片驗證，把「訓練串接」這條也打通。它是階梯 3 工業流程的教學縮小版。

---

## 2. 標準 pipeline（七層，三階梯共用）

```
[接入 Ingest]   YouTube(yt-dlp) / RTSP / 影片檔        ← 階梯換來源，介面不變
      │
[閘門 Gate]     移動偵測(MOG2) + ROI Crop              ← 省算力、濾雜訊(樹葉飄動)
      │
[偵測 Detect]   YOLO Nano  ★插槽1：換權重             ← COCO / 自訓練 best.pt
      │
[追蹤 Track]    ByteTrack (維持 ID，防重複)
      │
[規則 Rule]     ★插槽2：換邏輯                         ← 分流 / 穿線計數
      │
[輸出 Sink]     ★插槽3：換出口                         ← console/csv / TG / MQTT
      │
[部署 Deploy]   Colab(開發) → RPi5 + NCNN(邊緣，最後階段)
```

四層共用（接入/閘門/追蹤/部署邏輯），三個插槽換件就切換階梯：

| 插槽 | 教學 | 居家 | 工業 |
|---|---|---|---|
| 1 偵測權重 | `yolo11n.pt`（COCO） | `yolo11n.pt`（COCO） | 自訓練 `best.pt`（合成+fine-tune） |
| 2 規則 | 無（純驗證）/ 計數 demo | event_router 分流 | line_crossing 計數 |
| 3 輸出 | console + 標註影片 | telegram + csv | mqtt/http + csv → MES |

程式碼落點：共用四層在 `core/`，插槽 2 在 `rules/`，插槽 3 在 `sinks/`，每個階梯一個 `profiles/*.yaml` 描述三個插槽怎麼配。

---

## 3. 雲端優先技術棧（不佔本地）

| 角色 | 工具 | 免費額度 | 為何選它 |
|---|---|---|---|
| 運算 | Google Colab（T4 GPU） | 每日數小時 | 免安裝、訓練/推論都在雲端 |
| 儲存 | Google Drive | 15GB | 影片/圖片/資料集/checkpoint 全放這，本地零佔用 |
| 取影片 | yt-dlp | — | 下載 YouTube 公開影片到 Drive |
| 標註 | Roboflow（免費層）/ CVAT | — | 資料不敏感用 Roboflow 最快；敏感自架 CVAT |
| 模型 | Ultralytics YOLO11 | 開源 | 偵測 + 內建 ByteTrack + NCNN 匯出 |
| 邊緣（最後） | RPi5 + NCNN | 一次性硬體 ~NT$2,000 | 現場推論，斷網可用 |

對齊既有課程的雲端優先精神：Colab 免安裝、雙軌可完成、HiveMQ/Streamlit Cloud 都走免費雲端層。

---

## 4. 部署與測試路徑（照階梯走，每步有驗收）

### 階梯 1 — 教學（純雲端，先做這個）
1. 開 `notebooks/tier1_teaching_youtube_yolo.ipynb`（Colab）
2. 掛 Google Drive → 設定一支 YouTube 公開影片網址（含目標物，如街景人車）
3. yt-dlp 下載影片到 Drive → YOLO COCO 跑偵測/追蹤 → 標註影片 + 各類別計數存回 Drive
4. **驗收**：標註影片中目標物被框出、信心度合理、各類別數量印出 → 辨識流程打通

半階（訓練串接，選做）：挑一個 COCO 沒有的標的 → Roboflow 標 50-100 張 → Colab T4 訓練 → 換新影片驗證 mAP/recall。

### 階梯 2 — 居家（換資料來源）
1. 確認 ipcam 能否取 RTSP（小米多半要 Micam 橋接 / 刷機 / 換機，先查型號晶片）
2. 同 pipeline 換 `profiles/home_sensing.yaml`（COCO + event_router + telegram）
3. 加移動閘門濾雜訊；人車存檔+TG 通知，動物只 log
4. **驗收**：真人經過觸發一次 TG（不重複洗版）、樹葉飄動不誤觸發

### 階梯 3 — 工業（自訓練 + 邊緣）
1. 收自家產品照片 + RTSP 空背景 → Copy-Paste 合成 3,000-5,000 張（CPU/Colab）
2. Colab T4 fine-tune YOLO nano（real + synthetic）→ 匯出 NCNN 到 Drive
3. 換 `profiles/factory_counting.yaml`（best.pt + line_crossing + mqtt）
4. 下載 NCNN 到 RPi5 → 設 ROI + 計數線 → Shadow Mode 與人工/PLC 對帳
5. **驗收**：Detection Recall >90%、Counting Error <10%、RPi5 ≥3-5 FPS、連續運行 >2hr

---

## 5. 邊界（先不做）

- 完整 Dashboard / MES 深整合 / 多相機融合（工業 P1+）
- IMX500 感測器內推論（要 .rpk 轉檔版本地獄，本架構推論放主機/Colab/RPi 更彈性）
- 雲端 MLOps、Stable Diffusion 全圖生成（合成優先用 Copy-Paste，更可控）

---

## 6. 隱私

RTSP 在區網、推論在 Colab(開發)或 RPi(現場)、原始影像只存 Drive/本地、對外只匯出去識別化的結構化事件（計數/告警）。符合個資法「個人影像不離開可控環境」的論述。
