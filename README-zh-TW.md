# Image Similarity Detector

**訓練前先檢查影像資料集：找出重複檔案、審查近似圖片、比對資料集切分。全程在本機處理。**

[English](README.md) · [CLI 說明](docs/USAGE.md) · [程式架構](docs/ARCHITECTURE.md) · [報告格式](docs/REPORT_SCHEMA.md)

![本機影像資料集審查報告](docs/assets/report.png)

## 專案定位

影像資料集可能混有重複匯出、重新壓縮的畫面，或出現在 train 與 test 的相關影像。本工具把結果整理成可審查的資料品質報告，協助你決定下一步。

專案起源於內部使用的小型腳本。0.2 版整理為有測試的 Python 套件與 CLI，新增分塊比對、跨資料夾檢查、快取與離線 HTML。目前是 **beta 資料集審查工具**，實際辨識效果須用你的資料與門檻驗證。

## 功能

- SHA-256 完全重複檢查，與視覺相似候選分開標示。
- 預設 256-bit 感知差異雜湊；可選 ResNet50 特徵。
- 使用命名資料夾，比對 train／validation／test。
- 分塊計算，避免配置完整 N × N 相似度矩陣。
- ResNet50 真正批次推論，支援 CPU、CUDA、Apple MPS。
- 依檔案內容與編碼器版本區分快取，損壞會重新計算。
- HTML 內嵌縮圖，支援類型篩選、路徑搜尋，不需網頁伺服器。
- JSON 有 schema 版本、來源資訊、完整統計與略過原因。
- 不刪除、移動或修改原始影像。

## 安裝與掃描

需要 Python 3.10 以上。CI 目標涵蓋 Linux、Windows、macOS。

```bash
git clone https://github.com/livejiaquan/image-similarity-detector.git
cd image-similarity-detector
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install .
image-similarity scan --input images=./dataset/images --output ./results
```

每次掃描建立獨立結果目錄，含 `report.html` 與 `report.json`。直接用瀏覽器開啟 HTML 即可。預設模式僅需 NumPy 與 Pillow，安裝後可離線使用。

### 檢查資料集切分

```bash
image-similarity scan \
  --input train=./dataset/train \
  --input test=./dataset/test \
  --scope cross-root \
  --cache-dir ./.image-similarity-cache \
  --output ./results
```

輸入名稱需唯一、資料夾不得重疊；輸出與快取放在輸入資料夾外。

### 試用合成示範

```bash
python examples/make_demo.py --output demo-data
image-similarity scan --input train=demo-data/train --input test=demo-data/test
```

示範是程式生成的插圖、完全相同副本與 JPEG 變體，不含公司影像。預設得到 8 張影像、2 對完全重複、4 對近似候選。這是流程驗證，不是實際資料集準確率。也可下載[示範 HTML](docs/demo/report.html)到本機開啟。

### ResNet50

```bash
python -m pip install ".[neural]"
image-similarity scan --input images=./dataset/images \
  --backend resnet50 --threshold 0.99 --batch-size 32 --device auto
```

首次需下載 ImageNet `IMAGENET1K_V2` 權重。CUDA 環境請依 [PyTorch 官方說明](https://pytorch.org/get-started/locally/)安裝相容套件。

## 解讀報告

| 標記 | 意義 |
| --- | --- |
| Exact | SHA-256 相同 |
| Near | 位元組不同，但視覺分數達門檻，需人工確認 |
| Cross-root | 來自不同命名輸入，需確認切分與來源 |
| Partial | 有無法讀取或略過的輸入，詳情列在報告 |

dHash 分數是 `1 − 不同比特數 / 256`，ResNet50 是餘弦相似度。分數不是信心機率，兩種門檻不可互換。dHash 可能忽略顏色與局部細節，低資訊圖片會碰撞；ResNet50 可能把相同主題的不同照片視為相似。

A 像 B、B 像 C，不代表 A 與 C 可以互相取代。因此僅對完全相同檔案分組，不推論「可以安全刪除幾張」。跨集合相似也不能直接斷言資料洩漏。

所有符合範圍的圖片對都會比較。預設 JSON 保留前 10,000 對匹配，HTML 顯示前 100 對。總數、截斷與顯示限制都有標示；固定分塊順序並非最高分排名。

## 限制與資料處理

分塊降低相似度矩陣記憶體需求，計算量仍是 **O(N²)**。特徵矩陣隨影像數線性增長；目前沒有近似索引，也不宣稱支援百萬張影像。

支援 JPEG、PNG、BMP、GIF、WebP、TIFF。EXIF 方向正規化；動畫與多頁只取第一幀。壞圖與符號連結會略過並記錄。任何輸入資料夾沒有可讀取的影像（包括空資料夾、只有不支援格式的檔案），掃描都會標為 Partial；加上 `--strict` 會以退出碼 2 失敗，避免缺少某個資料集切分卻被當成檢查通過。未使用 `--strict` 時，解讀零匹配前請先確認報告狀態。

報告含縮圖與檔名，JSON 另含絕對路徑。分享前請確認對象；快取及結果預設不進 Git。公開示範僅用合成資料。

## 開發

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
ruff format --check src tests examples
python -m pytest
python -m build
```

程式依 discovery、features、cache、matching、pipeline、reporting 分工。CI 包含核心測試、神經網路架構檢查及 wheel 安裝驗證。已執行檢查見 [VERIFICATION.md](docs/VERIFICATION.md)。

[參與貢獻](CONTRIBUTING.md) · [版本變更](CHANGELOG.md) · [MIT](LICENSE)
