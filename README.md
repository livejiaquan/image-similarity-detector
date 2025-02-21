# 📌 Image Similarity Detector

## **1. Overview**

**Image Similarity Detector** is a tool designed for detecting similar images in a dataset using deep learning-based feature extraction. It leverages a pre-trained **ResNet50** model to extract features from images and computes their cosine similarity. The tool is useful for tasks such as:

- **Duplicate Image Detection**: Identify and remove duplicate or near-duplicate images.
- **Dataset Cleaning**: Ensure datasets do not contain redundant images before training.
- **Surveillance Footage Analysis**: Detect repeated frames in CCTV footage.

The tool supports **cross-folder comparison**, **in-folder duplicate detection**, and outputs similarity reports along with an optional visualization of similar image pairs.

---

## **2. Features**

- **Feature Extraction**: Uses a **ResNet50** model to extract 2048-dimensional feature vectors.
- **Similarity Computation**: Cosine similarity is used to measure image similarity.
- **Threshold-Based Filtering**: Users can define a similarity threshold (default: `0.99`).
- **Cross-Folder or Same-Folder Comparison**: Users can choose to compare across different directories or only within the same directory.
- **JSON Report Generation**: Stores results in structured JSON format.
- **Image Pair Collage Generation**: Visualizes similar image pairs.
- **Optimized Processing**: Uses batch processing for efficient feature extraction.

---

## **3. Installation**

### **📌 3.1 Prerequisites**
Ensure the following dependencies are installed:

```bash
pip install torch torchvision numpy tqdm opencv-python pillow
```

---

## **4. Usage**

### **📌 4.1 Running the Tool**
The script can be executed with the default settings:

```bash
python image_similarity_detector.py
```

To customize parameters, modify the `main()` function inside `image_similarity_detector.py`.

### **📌 4.2 Configuration Parameters**
Users can adjust the following parameters before execution:

| Parameter        | Description                                          | Default Value |
|-----------------|------------------------------------------------------|--------------|
| `folder_list`   | List of folders to scan for images                  | `['data/images/']` |
| `threshold`     | Similarity threshold for detecting duplicates       | `0.99` |
|                 | **Note**: In **static CCTV environments**, where the background remains largely unchanged, a **threshold of `0.99` is recommended** to detect only highly similar images. |
| `compare_mode`  | Comparison mode: `full` (all images) or `sample` (subset) | `full` |
| `sample_ratio`  | If using `sample` mode, defines sample proportion  | `0.3` (30%) |
| `compare_scope` | `all` (compare all images) or `inter_folder` (exclude same-folder comparisons) | `all` |
| `max_pairs_for_collage` | Maximum number of image pairs to visualize in collage | `20` |

---

## **5. Output Results**

### **📌 5.1 Console Output Example**

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

### **📌 5.2 Output Directory Structure**
After execution, results will be saved in a structured format:

```
results/
│— 20250219_123456/  ← (Current execution output)
│   ├─ report.json  ← (JSON file containing similarity results)
│   ├─ similar_pairs_sample/  
│   │   ├─ random_collage.jpg  ← (Collage of similar image pairs)
```

### **📌 5.3 JSON Report Example**
```json
{
    "timestamp": "20250219_123456",
    "folder_list": ["data/images"],
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
This JSON file contains details of all detected similar image pairs, including their paths and similarity scores.

---

## **6. Development and Contribution**

### **📌 6.1 Project Structure**
```
image_similarity_detector.py  # Main script
results/                      # Output directory
README.md                     # Documentation
```

### **📌 6.2 Future Enhancements**
- Add support for other feature extraction models (e.g., MobileNet, EfficientNet).
- Implement GUI for easier usage.
- Optimize batch processing for large datasets.

---

## **7. Summary**
✅ Extracts image features using **ResNet50**.
✅ Computes **cosine similarity** between images.
✅ Identifies and **visualizes duplicate image pairs**.
✅ Generates **detailed JSON reports**.
✅ Supports **cross-folder and same-folder comparisons**.

This tool provides an efficient way to manage large image datasets by identifying and filtering duplicate or highly similar images. For any feature requests or issues, feel free to contribute via GitHub!

