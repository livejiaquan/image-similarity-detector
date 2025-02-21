# 📌 圖片相似度檢測工具使用說明

## **1. 工具介紹**
本工具用於 **檢測資料夾中相似的圖片**，並生成報告。

🔹 **核心技術**：
- 使用 **ResNet50** 提取 **2048 維特徵向量**。
- 計算 **Cosine Similarity 相似度矩陣**。
- 設定閾值篩選 **相似圖片對**。
- **可選**：跨資料夾比對 or 內部比對。
- 產出 **報告 + 拼接圖片**。

🔹 **應用場景**：
✅ 找出 **重複圖片**（去重 Deduplication）
✅ **監控影像**（CCTV 重複畫面偵測）
✅ **標註前檢查數據集**（確保不標記相同圖片）
✅ **減少存儲空間**（刪除重複圖片）

---

## **2. 使用方式**

### **📌 2.1 設定參數（可在 `main()` 內調整）**

| 參數名稱 | 說明 | 預設值 |
|----------|---------------------------------|-------|
| `folder_list` | 要掃描的圖片資料夾列表 | `['CCTVforSora/']` |
| `threshold` | 相似度閾值（越高越嚴格）| `0.99` |
| | **說明**：如果是在 **廠區靜態 CCTV 場景**，背景大部分相同，可以設置為 **0.99** 以確保只偵測出極為相似的影像。 |
| `compare_mode` | 比對模式 (`full` or `sample`) | `full` |
| `sample_ratio` | 若 `sample` 模式，設定取樣比例 | `0.3` (30%) |
| `compare_scope` | `all` (全部比對) 或 `inter_folder` (跨資料夾比對) | `all` |
| `max_pairs_for_collage` | 生成拼接圖時，最多挑選多少組相似對 | `20` |

---

### **📌 2.2 執行程式**
確保 Python 環境已安裝 PyTorch，然後執行：
```bash
python image_similarity_detector.py
```

---

## **3. 輸出結果**

### **📌 3.1 終端機輸出範例**
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

### **📌 3.2 產出報告結構**
結果會存到 `results/` 目錄，格式如下：
```
results/
│── 20250219_123456/  ← (本次運行的輸出)
│   ├── report.json  ← (包含相似圖片資訊的 JSON)
│   ├── similar_pairs_sample/  
│   │   ├── random_collage.jpg  ← (隨機挑選相似圖片的拼接圖)
```

### **📌 3.3 JSON 報告格式範例**
```json
{
    "timestamp": "20250219_123456",
    "folder_list": ["CCTVforSora"],
    "total_images_in_folders": 1500,
    "compare_mode": "full",
    "threshold": 0.99,
    "number_of_similar_pairs": 235,
    "number_of_unique_groups": 1200,
    "photos_to_remove": 300,
    "similar_pairs": [
        {
            "index": 1,
            "image1": "CCTVforSora/img1.jpg",
            "image2": "CCTVforSora/img2.jpg",
            "similarity_score": 0.9956
        },
        {
            "index": 2,
            "image1": "CCTVforSora/img3.jpg",
            "image2": "CCTVforSora/img4.jpg",
            "similarity_score": 0.9923
        }
    ]
}
```
這份報告詳細記錄了：
- **找到的相似圖片對**
- **它們的相似度分數**
- **如果要刪除重複圖片，最多能刪除多少**

---

## **4. 總結**
這個工具使用 **ResNet50 提取圖片特徵**，然後用 **Cosine Similarity** 來比較圖片的相似度，最後生成 **報告 + 拼接圖片**，幫助使用者快速檢查相似圖片。

✨ **這樣的說明清楚嗎？如果有任何問題，歡迎詢問！😊**