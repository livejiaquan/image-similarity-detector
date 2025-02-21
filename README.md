# 📌 圖片相似度檢測工具

## **1. 專案簡介**
本工具基於 **ResNet50 深度學習模型**，可用於檢測資料夾內的 **相似圖片**，適用於去重 (Deduplication)、監控影像比對、數據集標註檢查等場景。

## **2. 主要功能**
- **圖片特徵提取**：使用 ResNet50 進行特徵向量抽取。
- **相似度計算**：基於 **Cosine Similarity** 計算圖片相似度。
- **相似圖片篩選**：使用 **閾值 (Threshold)** 過濾相似圖片對。
- **比對模式**：
  - `all`：比對所有圖片。
  - `inter_folder`：僅進行跨資料夾比對。
- **輸出結果**：生成 **相似圖片對 JSON 報告**，並 **可選擇拼接圖片輸出**。

## **3. 使用方式**

### **📌 3.1 設定參數（於 `main()` 內調整）**

| 參數名稱 | 說明 | 預設值 |
|----------|---------------------------------|-------|
| `folder_list` | 欲掃描的圖片資料夾 | `['CCTVforSora/']` |
| `threshold` | **相似度閾值** (值越高匹配越嚴格) | `0.99` |
| | **說明**：若為 **廠區靜態 CCTV 場景** (背景相同)，可設置 **0.99**，確保僅偵測高度相似圖片。 |
| `compare_mode` | 比對模式 (`full` or `sample`) | `full` |
| `sample_ratio` | `sample` 模式下的抽樣比例 | `0.3` (30%) |
| `compare_scope` | `all` (全部比對) or `inter_folder` (跨資料夾比對) | `all` |
| `max_pairs_for_collage` | 生成拼接圖時，最多顯示幾組 | `20` |

### **📌 3.2 執行程式**
```bash
python main.py
```

## **4. 輸出結果**

### **📌 4.1 終端機輸出範例**
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

### **📌 4.2 產出報告結構**
```
results/
│── 20250219_123456/  ← (本次運行輸出)
│   ├── report.json  ← (相似圖片 JSON 報告)
│   ├── similar_pairs_sample/  
│   │   ├── random_collage.jpg  ← (隨機相似圖片拼接圖)
```

### **📌 4.3 JSON 報告格式範例**
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

此報告詳細記錄了：
- **找到的相似圖片對**
- **對應相似度分數**
- **若刪除重複圖片，最多可刪除數量**

## **5. 主要技術細節**

### **📌 5.1 圖片特徵提取**
- **模型**：ResNet50 (移除全連接層，提取 **2048 維特徵向量**)
- **預處理**：
  - 調整大小至 `(224, 224)`
  - 轉換為 `Tensor`
  - **標準化 (Normalization)**：
    ```python
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ```

### **📌 5.2 相似度計算**
- 使用 **Cosine Similarity** 計算圖片特徵之間的相似度。
- 公式：
  $$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- **特徵矩陣計算範例**：
  ```python
  norms = np.linalg.norm(features, axis=1, keepdims=True)
  normalized = features / (norms + 1e-10)
  similarity_matrix = np.dot(normalized, normalized.T)
  ```

### **📌 5.3 Union-Find 算法 (連通分量)**
- **用途**：確保圖片只保留一份，去除多餘重複圖片。
- **方法**：
  ```python
  uf = UnionFind(n_selected)
  for (p1, p2, sim) in duplicates:
      i = path2idx[p1]
      j = path2idx[p2]
      uf.union(i, j)
  ```
- **計算可刪除圖片數量**：
  ```python
  photos_to_remove = n_selected - uf.count
  ```

## **6. 總結**
✅ **高效 ResNet50 圖片特徵提取**，適用於大規模數據。
✅ **Cosine Similarity** 確保圖片相似性計算準確。
✅ **閾值可調整** (`0.99` 適用於靜態場景，如 CCTV 監控)。
✅ **產生 JSON 報告**，方便後續數據分析與處理。
✅ **支援拼接輸出相似圖片對**，直觀查看結果。

📌 **適用場景**：監控影像比對、數據集標註檢查、重複圖片清理等。

🚀 若有任何問題或改進建議，請隨時提交 Issue 或 Pull Request！

