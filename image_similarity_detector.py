import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json
import random
from datetime import datetime

###########################
# Union-Find (Disjoint Set)
###########################
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.count = n  # 連通分量數量

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            # 將 rank 較低的樹合併到 rank 較高的樹
            if self.rank[rootx] < self.rank[rooty]:
                self.parent[rootx] = rooty
            elif self.rank[rootx] > self.rank[rooty]:
                self.parent[rooty] = rootx
            else:
                self.parent[rooty] = rootx
                self.rank[rootx] += 1
            self.count -= 1  # 合併一次，就少一個連通分量

def get_image_paths(folder_list):
    """
    根據傳入的資料夾清單，收集所有圖片檔的路徑，
    並回傳 (image_paths, folder_map)。
    - image_paths: list[str]，所有圖片的絕對路徑
    - folder_map: dict[str, str]，key=圖片絕對路徑, value=該圖片隸屬的資料夾路徑
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    folder_map = {}
    for folder in folder_list:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in exts:
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
                    folder_map[full_path] = folder
    return image_paths, folder_map

class ImageFeatureExtractor:
    """
    圖片特徵抽取器，使用預訓練的 ResNet50 並只取最後一層全連接層前的特徵向量。
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.resnet50(pretrained=True)
        # 移除最後一層 (fc layer)，只保留特徵部分
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # 圖片的標準預處理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path):
        """
        將輸入圖片路徑讀進來並轉成特徵向量 (1, 2048) 的 ndarray。
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)  # shape=(1,2048,1,1)
        features = features.view(features.size(0), -1)  # shape=(1,2048)
        return features.cpu().numpy().astype(np.float32)

def compute_feature_matrix(image_paths, extractor):
    """
    給定圖片路徑列表和特徵抽取器，批次處理並回傳 shape=(N, feature_dim) 的特徵矩陣。
    """
    all_features = []
    for path in tqdm(image_paths, desc="Extracting features"):
        feat = extractor.extract(path)
        all_features.append(feat[0])  # shape=(1,2048)
    return np.vstack(all_features)

def compute_similarity_matrix(features):
    """
    計算 cosine similarity 矩陣 (N,N)，範圍[-1,1]。
    features: shape=(N,2048)
    回傳 similarity_matrix: shape=(N,N)
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / (norms + 1e-10)
    return np.dot(normalized, normalized.T)

def find_duplicates(similarity_matrix, image_paths, threshold=0.95,
                    folder_map=None, compare_scope='all'):
    """
    在相似度矩陣中，找出 sim >= threshold 的圖片對 (i<j)。
    - compare_scope='all': 不分資料夾，全比
    - compare_scope='inter_folder': 不比同資料夾內圖片
      (若 folder_map[path_i] == folder_map[path_j] 即跳過)
    回傳 duplicates: list of (path_i, path_j, sim_score)
    """
    duplicates = []
    N = similarity_matrix.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if compare_scope == 'inter_folder' and folder_map is not None:
                # 如果兩張圖來自同一個資料夾，就跳過
                if folder_map[image_paths[i]] == folder_map[image_paths[j]]:
                    continue
            sim_score = similarity_matrix[i, j]
            if sim_score >= threshold:
                duplicates.append((image_paths[i], image_paths[j], float(sim_score)))
    return duplicates

def create_pair_image_in_memory(path1, path2, sim_score=None, pair_height=200):
    """
    用 OpenCV 讀取兩張圖並排，最後統一縮放到 pair_height 高度。
    回傳: np.ndarray (BGR) 或 None
    """
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    common_h = min(h1, h2)

    def resize_keep_aspect(img, new_h):
        hh, ww = img.shape[:2]
        ratio = new_h / hh
        new_w = int(ww * ratio)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    r1 = resize_keep_aspect(img1, common_h)
    r2 = resize_keep_aspect(img2, common_h)
    combined = np.concatenate((r1, r2), axis=1)

    if sim_score is not None:
        cv2.putText(
            combined,
            f"Sim: {sim_score:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    # 將並排後的圖縮放到 pair_height
    hh, ww = combined.shape[:2]
    if hh > 0:
        ratio = pair_height / hh
        new_w = int(ww * ratio)
        combined = cv2.resize(combined, (new_w, pair_height), interpolation=cv2.INTER_LANCZOS4)
    return combined

def create_collage_from_pairs(pair_images, pairs_per_row=3):
    """
    將所有 pair_images 拼接成一張大圖 (collage)。
    同一行拼接後若寬度不一，則補白到最大的寬度，再做垂直串接。
    回傳: collage (np.ndarray) 或 None
    """
    if not pair_images:
        return None

    rows = []
    for i in range(0, len(pair_images), pairs_per_row):
        chunk = pair_images[i : i+pairs_per_row]
        row_img = np.concatenate(chunk, axis=1)
        rows.append(row_img)

    max_width = max(r.shape[1] for r in rows)
    padded_rows = []
    for row_img in rows:
        h, w, c = row_img.shape
        if w < max_width:
            pad_w = max_width - w
            pad = np.full((h, pad_w, 3), [255,255,255], dtype=np.uint8)
            row_padded = np.concatenate((row_img, pad), axis=1)
        else:
            row_padded = row_img
        padded_rows.append(row_padded)

    collage = np.concatenate(padded_rows, axis=0)
    return collage

def main():
    # ========== 參數設定 ==========
    folder_list = [
        r"/--You Dataset Path--",
        # r"--You Dataset Path--"
    ]
    threshold = 0.99
    compare_mode = 'full'   # 'full' or 'sample'
    sample_ratio = 0.3
    compare_scope = 'all'   # 'all' or 'inter_folder'
    max_pairs_for_collage = 20

    # 輸出資料夾
    results_base_dir = "results"
    os.makedirs(results_base_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_result_dir = os.path.join(results_base_dir, timestamp_str)
    os.makedirs(run_result_dir, exist_ok=True)

    # ========== 取得圖片路徑 + 資料夾紀錄 ==========
    image_paths, folder_map = get_image_paths(folder_list)
    total_images = len(image_paths)
    print(f"Total images found: {total_images}")
    if total_images == 0:
        print("No images found. Terminated.")
        return

    # ========== 抽樣或全量 ==========
    if compare_mode == 'sample':
        sample_size = max(1, int(total_images * sample_ratio))
        selected_paths = random.sample(image_paths, sample_size)
        print(f"Sampling mode: from {total_images} images, sampling {sample_size}.")
    else:
        selected_paths = image_paths

    # ========== 抽取特徵 + 計算相似度 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    extractor = ImageFeatureExtractor(device=device)

    features = compute_feature_matrix(selected_paths, extractor)
    similarity_matrix = compute_similarity_matrix(features)

    # ========== 找相似對 ==========
    duplicates = find_duplicates(
        similarity_matrix,
        image_paths=selected_paths,
        threshold=threshold,
        folder_map=folder_map,
        compare_scope=compare_scope
    )
    num_duplicates = len(duplicates)
    print(f"Threshold = {threshold}")
    print(f"Found {num_duplicates} pairs of images that meet the similarity threshold.")

    # ========== (新增) Union-Find：計算可刪除數量 ==========
    # 1) 建立 <path -> index> map
    path2idx = {p: idx for idx, p in enumerate(selected_paths)}
    # 2) 初始化 UnionFind
    n_selected = len(selected_paths)
    uf = UnionFind(n_selected)
    # 3) 逐對 duplicates union
    for (p1, p2, sim) in duplicates:
        i = path2idx[p1]
        j = path2idx[p2]
        uf.union(i, j)
    # 4) 獨立群組數 = uf.count
    number_of_unique_groups = uf.count
    # 可刪除張數 = n_selected - number_of_unique_groups
    photos_to_remove = n_selected - number_of_unique_groups

    print(f"Number of unique groups: {number_of_unique_groups}")
    print(f"Photos that can be removed if only keeping 1 from each group: {photos_to_remove}")

    # ========== JSON 報告準備 ==========
    result_info = {
        "timestamp": timestamp_str,
        "folder_list": folder_list,
        "total_images_in_folders": total_images,
        "compare_mode": compare_mode,
        "sample_ratio": sample_ratio if compare_mode == 'sample' else None,
        "compare_scope": compare_scope,
        "threshold": threshold,
        "number_of_images_used_for_comparison": n_selected,
        "number_of_similar_pairs": num_duplicates,
        "number_of_unique_groups": number_of_unique_groups,
        "photos_to_remove": photos_to_remove,
        "similar_pairs": []
    }

    for idx, (p1, p2, sim_score) in enumerate(duplicates):
        result_info["similar_pairs"].append({
            "index": idx+1,
            "image1": p1,
            "image2": p2,
            "similarity_score": sim_score
        })

    # ========== 拼圖 (隨機取部分對) ==========
    if num_duplicates > 0:
        duplicates_dir = os.path.join(run_result_dir, "similar_pairs_sample")
        os.makedirs(duplicates_dir, exist_ok=True)

        if num_duplicates > max_pairs_for_collage:
            random_subset = random.sample(duplicates, max_pairs_for_collage)
        else:
            random_subset = duplicates

        pair_images = []
        for (p1, p2, sim_score) in random_subset:
            pair_img = create_pair_image_in_memory(p1, p2, sim_score=sim_score, pair_height=200)
            if pair_img is not None:
                pair_images.append(pair_img)

        collage = create_collage_from_pairs(pair_images, pairs_per_row=3)
        if collage is not None:
            collage_path = os.path.join(duplicates_dir, "random_collage.jpg")
            cv2.imwrite(collage_path, collage)
            print(f"Collage saved: {collage_path}")
            result_info["sample_collage_path"] = os.path.abspath(collage_path)
        else:
            print("Warning: could not create collage from random subset (no valid images).")
    else:
        print("No similar pairs => No collage generated.")

    # ========== 寫出 JSON ==========
    json_path = os.path.join(run_result_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_info, f, indent=4, ensure_ascii=False)

    print(f"\nResult JSON: {json_path}")
    print(f"Done. Output folder: {run_result_dir}")

if __name__ == "__main__":
    main()
