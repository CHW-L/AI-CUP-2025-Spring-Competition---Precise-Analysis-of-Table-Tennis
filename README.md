# AI-CUP-2025-Spring-Competition---Precise-Analysis-of-Table-Tennis 
> **單一 Python 腳本 ‧ 端到端特徵工程 + LightGBM 基線**

---

## 1. 環境配置
| 類別 | 規格 / 版本 |
|------|-------------|
| 作業系統 | **Windows 10 x64** |
| 開發/訓練硬體 | Intel Core i7-8750H ‧ 32 GB RAM ‧ NVIDIA GeForce GTX 1060 |
| Python | **3.11.4** |
| 主要套件 | `numpy 1.25.2` ‧ `pandas 2.0.3` ‧ `scipy 1.15.3` ‧ `scikit-learn 1.6.1` ‧ `lightgbm 4.6.0` ‧ `tqdm 4.67.1` ‧ `re 2.2.1` *(標準函式庫)* |
| 外部資料 / 預訓練模型 | **皆未使用** ‒ 所有特徵即時計算自官方六軸 IMU 原始訊號 |
| 可重現性 | 程式內固定隨機種子，確保結果一致 |

---

## 2. 專案結構與程式碼路徑
.
├── 39_Training_Dataset/
│ └── 39_Training_Dataset/
│ ├── train_info.csv
│ └── train_data/unique_id.txt
├── 39_Test_Dataset/
│ └── 39_Test_Dataset/
│ ├── test_info.csv
│ └── test_data/unique_id.txt
└── AI CUP 2025 TEAM_7541.py ← 唯一腳本 (本隊提交)
> *路徑請與上表一致；若更名，請同步修改 `AI CUP 2025 TEAM_7541.py` 內的 `dir_train`／`dir_test`。*

---

## 3. 腳本功能摘要
| 階段       | 內容                                                              |
| -------- | --------------------------------------------------------------- |
| **特徵工程** | 27 段切片 × (8 軸 × 12 統計 + FFT 6 維 + 時長 31 維 + mode 1 維) → 2 646 維 |
| **模型訓練** | LightGBM；StratifiedGroupKFold 交叉驗證 (5 fold，但類別過少將自動降至 2–4 fold) |
| **預測集成** | 同任務多fold平均；多分類任務機率重分配 (自訂 `rr()` 函式)  |
| **輸出**   | `submission_f.csv`：`unique_id` + 4 任務(n分類有n機率欄位, n>=3) |

---
