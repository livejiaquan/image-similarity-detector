# 📌 圖片相似度檢測工具

## **1. 概述**

**圖片相似度檢測工具（Image Similarity Detector）** 是一款基於深度學習的工具，用於檢測數據集中相似的圖片。該工具利用 **ResNet50** 預訓練模型提取圖像特徵，並計算它們的餘弦相似度。適用於以下場景：

- **重複圖片檢測**：識別並刪除數據集中的重複或幾乎相同的圖片。
- **數據清理**：確保數據集中不包含冗餘圖片，提升數據質量。
- **監控影像分析**：檢測 **CCTV** 監控畫面中的重複場景。

該工具支持 **跨文件夾比對** 和 **同文件夾內部比對**，並提供相似度報告與可視化圖片。

---

## **2. 功能特點**

- **特徵提取**：使用 **ResNet50** 模型提取 **2048 維度特徵向量**。
- **相似度計算**：基於 **餘弦相似度（Cosine Similarity）** 測量圖片相似程度。
- **閾值篩選**：用戶可設置相似度閾值（預設 `0.99`）。
- **靈活比對範圍**：支持跨文件夾比對，或限制比對範圍至單一文件夾內。
- **JSON 報告輸出**：生成包含相似圖片詳細資訊的 JSON 報告。
- **圖片對拼接輸出**：可視化相似圖片對。
- **高效計算**：採用 **批量處理** 方式，加速特徵提取過程。

---

## **3. 安裝**

### **📌 3.1 依賴環境**
請確保已安裝以下 Python 套件：

```bash
pip install torch torchvision numpy tqdm opencv-python pillow
```

---

## **4. 使用方法**

### **📌 4.1 執行工具**

使用預設參數運行工具：
```bash
python image_similarity_detector.py
```

如需自定義參數，請修改 `image_similarity_detector.py` 內的 `main()` 函數。

### **📌 4.2 配置參數**

| 參數名稱        | 描述                                            | 預設值 |
|---------------|---------------------------------|------|
| `folder_list` | 需掃描的圖片文件夾列表                    | `['data/images/']` |
| `threshold`   | 相似度閾值，數值越高篩選越嚴格             | `0.99` |
|               | **說明**：在 **靜態 CCTV 監控場景**（背景基本不變）下，建議設定 **0.99** 來確保只檢測高度相似圖片。 |
| `compare_mode`| 比對模式：`full`（全數據比對）或 `sample`（抽樣） | `full` |
| `sample_ratio`| 若使用 `sample` 模式，設定抽樣比例         | `0.3` (30%) |
| `compare_scope`| `all`（比對所有圖片）或 `inter_folder`（排除同文件夾內比對） | `all` |
| `max_pairs_for_collage` | 生成相似圖片拼接時的最大圖片對數 | `20` |

---

## **5. 輸出結果**

### **📌 5.1 終端輸出範例**

```bash
Total images found: 1500
Threshold = 0.99
Found 235 pairs of images that meet the similarity threshold.
Number of unique groups: 1200
Photos that can be removed if only keeping 1 from each group: 300
Collage saved: results/20250219_123456/similar_pairs_sample/random_collage.jpg
Result JSON: results/20250219_123456/report.json
Done. Output folder: results/20250219_123456
```

### **📌 5.2 輸出目錄結構**

執行後，結果將存儲在 `results/` 目錄下：
```
results/
│── 20250219_123456/  ← (本次運行輸出)
│   ├── report.json  ← (包含相似圖片數據的 JSON 報告)
│   ├── similar_pairs_sample/  
│   │   ├── random_collage.jpg  ← (相似圖片對的拼接圖)
```

### **📌 5.3 JSON 報告範例**

```json
{
    "timestamp": "20250219_123456",
    "folder_list": ["data/images/"],
    "total_images_in_folders": 1500,
    "compare_mode": "full",
    "threshold": 0.99,
    "number_of_similar_pairs": 235,
    "number_of_unique_groups": 1200,
    "photos_to_remove": 300,
    "similar_pairs": [
        {
            "index": 1,
            "image1": "data/images/img1.jpg",
            "image2": "data/images/img2.jpg",
            "similarity_score": 0.9956
        },
        {
            "index": 2,
            "image1": "data/images/img3.jpg",
            "image2": "data/images/img4.jpg",
            "similarity_score": 0.9923
        }
    ]
}
```

該 JSON 文件詳細記錄了所有被檢測出的相似圖片對，包括圖片路徑與相似度分數。

---

## **6. 開發與貢獻**

### **📌 6.1 項目結構**
```
image_similarity_detector.py  # 主程式
results/                      # 輸出結果目錄
README.md                     # 文件說明
```

### **📌 6.2 未來改進方向**
- 支持其他特徵提取模型（如 MobileNet、EfficientNet）。
- 增加 **GUI 圖形化界面**，提升易用性。
- 優化 **大數據集** 處理能力，提高運行效率。

---

## **7. 總結**
✅ 使用 **ResNet50** 提取圖片特徵。
✅ 基於 **餘弦相似度** 計算圖片相似度。
✅ 檢測並 **可視化相似圖片對**。
✅ 生成 **詳細 JSON 報告**。
✅ 支持 **跨文件夾及同文件夾內比對**。

此工具提供了一種高效的方式來管理大型圖片數據集，通過識別並過濾重複或高度相似的圖片，確保數據集的高質量。如果有任何功能請求或問題，請通過 GitHub 進行反饋與貢獻！

