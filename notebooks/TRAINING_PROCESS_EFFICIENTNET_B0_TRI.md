# Bộ Quy Trình Huấn Luyện & Kiểm Thử Mô Hình `EfficientNet-B0` (Plant Disease ML — `data_v2`)

> **Tác giả:** Võ Minh Trí (`Tri`)  
> **Nhánh Git:** `feat/efficientnet-b0-tri`  
> **Bộ dữ liệu chính thức:** `data_v2` (`spMohanty/PlantVillage-Dataset`, chia lại theo nhóm từng chiếc lá gốc từ `leaf-map.json`, `split_fingerprint: 4fc4a896d43ddbfa...`, **0% rò rỉ**, đủ **38/38 lớp** ở cả 3 tập `train: 38.039` / `valid: 8.147` / `test: 8.098`)  
> **Phần cứng thực nghiệm:** NVIDIA GeForce RTX 3050 Laptop GPU (4.096 MiB VRAM, TGP 40 W), SSD NVMe Micron 2400, Python 3.12.16, PyTorch `2.14.1+cu126`  
> **Tài liệu nền tảng áp dụng:** *Designing Machine Learning Systems* (Chip Huyen, Chương 3–8) & *Hands-On Machine Learning* (Aurélien Géron, Chương 13–14)

---

## 1. Mục Tiêu & Tổng Quan 2 Phiên Bản Mô Hình

Toàn bộ quy trình giữ nguyên 100% các module dùng chung của nhóm (`notebooks/data_pipeline.py`, `notebooks/eval_engine.py`, `backend/`, `frontend/`) và huấn luyện 2 phiên bản mô hình trên bộ **`data_v2` sạch (`0%` rò rỉ theo chiếc lá, độ mất cân bằng thực tế `32,1×` từ `120` ảnh `Potato___healthy` tới `3.849` ảnh `Orange___HLB`)**:

1. **Phiên bản 1 (`models/efficientnet_b0_tri.pth`):** Tối ưu chuẩn thi đấu nhóm — đạt **`99,59%` Macro-F1** (`99,68%` Acc) trên `8.147` ảnh valid và **`99,35%` Macro-F1** (`99,62%` Acc, chỉ 31 lỗi) khi chấm **1 lần duy nhất** trên `8.098` ảnh held-out `data/test/`.
2. **Phiên bản 2 (`models/efficientnet_b0_robust_tri.pth`):** Bổ sung tiền xử lý ảnh điện thoại (`MobileLeafPreprocessor`: **Bilateral Filter + LAB-CLAHE + Adaptive Unsharp Mask**) đồng nhất giữa lúc train và suy luận (chống *Training-Serving Skew*) — vừa đạt **`99,12%` Macro-F1** (`99,37%` Acc) trên `8.098` ảnh held-out `data/test/`, vừa đẩy `Macro-F1` trên combo 4 nhiễu điện thoại (`combined_mobile`) từ **`27,92%` → `93,94%`** (**+66,02%**, giảm **`93,3%`** tổng số lỗi).

### Danh sách tệp trong nhánh `feat/efficientnet-b0-tri`

| Tệp | Vai trò |
| :--- | :--- |
| `notebooks/efficientnet_b0_tri.py` | Script huấn luyện **v1** (2 pha + AMP `fp16` + hỗ trợ `--resume_checkpoint` + đối chứng Baseline 1) |
| `notebooks/test_efficientnet_b0_tri.py` | 7 unit test TDD kiểm chứng kiến trúc, đóng băng/mở khóa pha 1–2, khớp 38 nhãn, tải `weights_only=True` & notebook |
| `notebooks/03_efficientnet_b0_tri.ipynb` | Notebook tổng hợp kết quả **v1**, biểu đồ hội tụ, telemetry GPU & held-out test `8.098` ảnh |
| `notebooks/efficientnet_b0_tri_results.json` | Nhật ký thực nghiệm chi tiết **v1** trên `data_v2` (15 epochs) |
| `notebooks/efficientnet_b0_tri_gpu_telemetry.csv` | 1.183 mẫu `nvidia-smi` đo thực tế suốt quá trình train trên RTX 3050 |
| `notebooks/data_v2_final_test_results_tri.json` | Kết quả chấm **1 lần duy nhất** trên tập held-out `data/test/` (`8.098` ảnh, 38 lớp) cho `v1`, `Robust v2` & `Baseline 1` |
| `notebooks/mobile_preprocessing_tri.py` | Bộ tiền xử lý `MobileLeafPreprocessor`, kiểm toán `0%` rò rỉ & 7 kịch bản Stress Test |
| `notebooks/test_mobile_preprocessing_tri.py` | 8 unit test TDD kiểm chứng LAB-CLAHE, làm nét, giảm nhiễu, tốc độ `< 5 ms/ảnh`, pickle Windows & `0%` rò rỉ `data_v2` |
| `notebooks/efficientnet_b0_robust_tri.py` | Script tinh chỉnh **v2 Robust** trên toàn bộ `38.039` ảnh `data_v2` + nhiễu điện thoại + `MobileLeafPreprocessor` |
| `notebooks/efficientnet_b0_robust_tri_results.json` | Nhật ký thực nghiệm **v2 Robust** trên `data_v2` |
| `notebooks/stress_test_comparison_tri.json` | Kết quả đối chứng Stress Test 4 cấu hình trên toàn bộ `8.147` ảnh valid `data_v2` (38/38 lớp) |
| `notebooks/04_mobile_preprocessing_and_stress_test_tri.ipynb` | Notebook minh họa tiền xử lý ảnh điện thoại & biểu đồ đối chứng Stress Test A/B/C |
| `models/efficientnet_b0_tri.pth` | Checkpoint **v1** (`15,76 MB`, nạp an toàn `weights_only=True`) |
| `models/efficientnet_b0_robust_tri.pth` | Checkpoint **Robust v2** (`15,77 MB`, nạp an toàn `weights_only=True`) |

---

## 2. Bước 1 — Kiểm Toán Rò Rỉ Dữ Liệu `data_v2` & Thiết Lập Môi Trường

### 2.1. Kết quả kiểm toán độc lập 4 tầng trên `data_v2` (`54.284` ảnh)
* **Khớp 38/38 lớp** với `models/class_indices.json` ở cả 3 tập `train (38.039)` / `valid (8.147)` / `test (8.098)`.
* **Ảnh hỏng:** `0` | **Trùng tên file:** `0` | **Trùng UUID / hậu tố xoay-lật Kaggle:** `0 / 8.147 (0,00%)` | **Trùng SHA-256 byte:** `0` (cả giữa 3 tập lẫn nội bộ từng tập).
* **Perceptual dHash bất biến xoay/lật:** Chỉ có `~0,02%` (vài cặp ngoại lệ chưa nằm trong `leaf-map.json`), sạch vượt trội so với bộ Kaggle cũ (rò rỉ `65,24%`).

### 2.2. Tạo môi trường ảo `.venv` (Python 3.12 + PyTorch CUDA)

```powershell
# Windows PowerShell (dùng uv, không cài vào Python toàn cục)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe scikit-learn pandas matplotlib seaborn pillow opencv-python-headless pytest ipykernel nbformat nbconvert jupyter

# Kiểm chứng GPU CUDA & OpenCV
.venv\Scripts\python.exe -c "import torch, cv2; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'OpenCV:', cv2.__version__)"
```

*Giải thích cờ lệnh nhanh:* `--python 3.12` chọn Python 3.12 tương thích wheel GPU; `--index-url .../whl/cu126` cài bản PyTorch biên dịch CUDA 12.6; `opencv-python-headless` cài OpenCV gọn nhẹ cho `Bilateral Filter` và `LAB-CLAHE`.

---

## 3. Bước 2 — Kiểm Thử Đơn Vị (TDD) & Tối Ưu Phần Cứng RTX 3050 (4 GB)

```powershell
# Chạy toàn bộ 15 unit test (bao gồm kiểm chứng 0% rò rỉ trên data/train & data/valid)
.venv\Scripts\pytest.exe -v
```

*Các tối ưu phần cứng đã kiểm chứng trên RTX 3050 Laptop (4 GB VRAM) chạy Windows:*
* **AMP `fp16` (`batch_size=32`):** Tăng tốc **+70,3%** (`285,3` vs `161,5 img/s`) và giảm **-47,8% VRAM** (`1.461` vs `2.801 MB`), giữ khoảng trống ~2,5 GB VRAM an toàn hơn nhiều so với `batch_size=64` (`2.845 MB`).
* **Tránh tràn pagefile Windows (`error 1455`):** Đặt `num_workers=2` (`persistent_workers=True`) cho `train_loader` và `num_workers=0` cho `valid_loader` giúp giảm 4× bộ nhớ commit ảo trên Windows mà vẫn duy trì tốc độ ~220 img/s.

---

## 4. Bước 3 — Huấn Luyện Phiên Bản 1 (`EfficientNet-B0 v1`)

### 4.1. Chiến lược & Lệnh chạy
* **Kiến trúc:** `torchvision.models.efficientnet_b0(weights=DEFAULT)` (`4.056.226` tham số), thay đầu ra bằng `Dropout(0.3) -> Linear(1280, 38)`.
* **Hàm mất mát:** `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)` — đặc biệt quan trọng trên `data_v2` vì độ lệch lớp thực tế là `32,1×`.
* **2 pha Transfer Learning:** Pha 1 Warm-up `3` epochs (`lr=1e-3`, khóa backbone) → Pha 2 Fine-tune `12` epochs (`backbone lr=1e-4`, `head lr=5e-4`, `CosineAnnealingLR`, `patience=4`).

```powershell
.venv\Scripts\python.exe notebooks/efficientnet_b0_tri.py `
  --epochs_phase1 3 `
  --epochs_phase2 12 `
  --batch_size 32 `
  --num_workers 2 `
  --patience 4 `
  --save_path "models/efficientnet_b0_tri.pth" `
  --history_out "notebooks/efficientnet_b0_tri_results.json"
```

### 4.2. Kết quả thực đo `v1` & `Robust v2` trên `data_v2` (`0%` rò rỉ, đủ 38/38 lớp)

| Mô hình / Tập đánh giá | Macro F1 (%) | Accuracy (%) | Weighted F1 (%) | Số ảnh sai | Batch Latency (ms/ảnh) | File `.pth` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline 1 (`SimplePlantCNN`) — Valid `8.147` | `0,06%` | `1,08%` | `0,02%` | — | `0,22 ms` | `1,64 MB` |
| **EfficientNet-B0 v1** — Valid `8.147` *(Best Ep 13)* | **`99,59%`** | **`99,68%`** | **`99,68%`** | `26 / 8.147` | `0,62 ms` (`1.626 FPS`) | `15,76 MB` |
| **EfficientNet-B0 v1 — Held-Out Test `8.098`** *(chấm 1 lần)* | **`99,35%`** | **`99,62%`** | **`99,62%`** | **`31 / 8.098`** | `0,30 ms` (đơn ảnh `21,46 ms`) | `15,76 MB` |
| **EfficientNet-B0 Robust v2** — Valid `8.147` *(Best Ep 2)* | **`99,39%`** | **`99,47%`** | **`99,47%`** | `43 / 8.147` | `0,62 ms` (`1.601 FPS`) | `15,77 MB` |
| **EfficientNet-B0 Robust v2 — Held-Out Test `8.098`** *(chấm 1 lần)* | **`99,12%`** | **`99,37%`** | **`99,37%`** | **`51 / 8.098`** | `0,30 ms` (đơn ảnh `21,05 ms`) | `15,77 MB` |

* **Telemetry GPU RTX 3050 (`1.183` mẫu `nvidia-smi` suốt 15 epochs v1):** Tải GPU (>5%) trung bình **`73,0%`**, trung vị **`83,0%`**, P95 **`96,0%`**; Peak VRAM PyTorch **`1.482,3 MB`** (`36,2%`), tổng hệ thống **`2.197 MiB`**; nhiệt độ tối đa **`87 °C`**, công suất TB **`29,33 W`**, tổng điện năng chỉ **`29,87 Wh`** (~0,03 kWh).

---

## 5. Bước 4 — Tiền Xử Lý Ảnh Điện Thoại & Huấn Luyện `Robust v2`

### 5.1. Lệnh tinh chỉnh `Robust v2` trên toàn bộ `38.039` ảnh `data_v2` + nhiễu điện thoại

```powershell
.venv\Scripts\python.exe notebooks/efficientnet_b0_robust_tri.py `
  --epochs 4 `
  --batch_size 32 `
  --num_workers 2 `
  --init_checkpoint "models/efficientnet_b0_tri.pth" `
  --save_path "models/efficientnet_b0_robust_tri.pth" `
  --history_out "notebooks/efficientnet_b0_robust_tri_results.json"
```

### 5.2. Bảng đối chứng Stress Test 4 cấu hình trên `8.147` ảnh valid `data_v2` (38/38 lớp)

| Kịch bản ảnh điện thoại | Mốc A: `v1` gốc Acc / F1 (Lỗi) | Mốc B: `v1` + Preproc *(Skew)* Acc / F1 (Lỗi) | Đối chứng: `v2` *(tắt Preproc)* Acc / F1 (Lỗi) | **Mốc C: `v2 Robust` + `MobileLeafPreprocessor`** Acc / F1 (Lỗi) | Cải thiện Mốc C vs Mốc A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clean`** *(Ảnh sạch)* | `99,68%` / `99,59%` (`26`) | `98,55%` / `97,94%` (`118`) | `99,24%` / `99,02%` (`62`) | **`99,47%` / `99,39%` (`43`)** | Giữ vững >99,3% |
| **`blur_mild`** *(Mờ nhẹ)* | `97,16%` / `95,80%` (`231`) | `93,56%` / `91,81%` (`525`) | `98,38%` / `98,07%` (`132`) | **`99,31%` / `99,17%` (`56`)** | **+2,15% Acc / +3,37% F1** |
| **`blur_heavy`** *(Mờ nặng)* | `68,81%` / `63,86%` (`2.541`) | `67,87%` / `63,83%` (`2.618`) | `95,32%` / `93,82%` (`381`) | **`98,65%` / `98,28%` (`110`)** | **+29,84% Acc / +34,42% F1** |
| **`harsh_lighting`** *(Tối râm)* | `97,13%` / `96,82%` (`234`) | `95,08%` / `94,29%` (`401`) | `97,05%` / `96,51%` (`240`) | **`99,21%` / `99,00%` (`64`)** | **+2,08% Acc / +2,18% F1** |
| **`sensor_noise`** *(Nhiễu hạt)* | `67,78%` / `63,55%` (`2.625`) | `61,97%` / `61,43%` (`3.098`) | `97,78%` / `96,85%` (`181`) | **`98,92%` / `98,60%` (`88`)** | **+31,14% Acc / +35,05% F1** |
| **`occlusion_22pct`** *(Che 22%)* | `97,79%` / `96,86%` (`180`) | `90,99%` / `88,79%` (`734`) | `97,35%` / `96,80%` (`216`) | **`98,27%` / `97,47%` (`141`)** | **+0,48% Acc / +0,61% F1** |
| **`combined_mobile`** *(Combo 4 nhiễu)* | **`31,74%` / `27,92%` (`5.561`)** | **`16,53%` / `17,34%` (`6.800`)** | `77,59%` / `73,63%` (`1.826`) | **`95,41%` / `93,94%` (`374`)** | **+63,67% Acc / +66,02% F1** *(giảm 93,3% lỗi)* |

---

## 6. Cách Chạy Lại Notebook & Nạp Checkpoint An Toàn

```powershell
# Chạy lại 2 notebook báo cáo
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/03_efficientnet_b0_tri.ipynb
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/04_mobile_preprocessing_and_stress_test_tri.ipynb
```

```python
import torch
from PIL import Image
from notebooks.efficientnet_b0_tri import build_efficientnet_b0
from notebooks.mobile_preprocessing_tri import get_mobile_eval_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/efficientnet_b0_robust_tri.pth", map_location=device, weights_only=True)
model = build_efficientnet_b0(num_classes=ckpt["num_classes"], pretrained=False).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Robust v2: use_preprocessor=True | v1 (models/efficientnet_b0_tri.pth) & PlantDoc: use_preprocessor=False
transform = get_mobile_eval_transforms(image_size=224, use_preprocessor=True)
img_tensor = transform(Image.open("duong_dan_anh_la.jpg").convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    probs = torch.softmax(model(img_tensor), dim=1)
```

---

## 7. Bước 5 — Fine-Tuning Gộp Chung `data_v2` + `data_plantdoc` 80/20

* **Nguồn dữ liệu:** Bộ chia chuẩn `data_plantdoc_80_20` của nhóm (`train: 1.871` ảnh / 28 lớp, `valid: 468` ảnh / 28 lớp, `test: 236` ảnh / 27 lớp; `0` ảnh hỏng; **`0` rò rỉ SHA-256 `train ↔ valid`**). Gộp chung với `data_v2/train` (`38.039` ảnh) thành **`39.910` ảnh** và tự động tạo cache cạnh dài tối đa `384px` cho tập train (`data/plantdoc_train_cache_384/`, giữ nguyên ảnh gốc `valid`/`test`) để rút thời gian huấn luyện xuống **`31,4 phút`**.
* **Lệnh fine-tune 2 giai đoạn từ `models/efficientnet_b0_tri.pth` (Stage 1: 55/45 + Stage 2: 75/25 + `min_pd_count=15` + CutMix/MixUp + Model EMA; chọn Best EMA thuần trên `PlantDoc Valid`, khóa kín `PlantDoc Test`):**

```powershell
.venv\Scripts\python.exe notebooks/efficientnet_b0_plantdoc_tri.py `
  --pv_data_dir "data" `
  --plantdoc_dir "data/plantdoc" `
  --init_checkpoint "models/efficientnet_b0_tri.pth" `
  --save_path "models/efficientnet_b0_plantdoc_tri.pth" `
  --history_out "notebooks/efficientnet_b0_plantdoc_tri_results.json" `
  --stage1_epochs 7 `
  --stage2_epochs 7 `
  --samples_per_epoch 10000 `
  --min_pd_count 15 `
  --train_cache_max_side 384
```

* **Kết quả chính thức (Best EMA Epoch 12/14 chọn trên Valid, Peak VRAM `1.499,4 MB`, `31,4 phút`, `23/23` unit tests PASSED):**
  * **PlantDoc Valid (`468` ảnh, 28 lớp):** Zero-Shot `17,52%` Top-1 / `16,43%` Macro-F1 → **`70,94%` Top-1 (`332/468`)**, **`89,10%` Top-3**, **`70,68%` Macro-F1** (`0,45 ms/ảnh` / `2.202,3 FPS`).
  * **PlantDoc Test (`236` ảnh, 27 lớp — khóa kín, chấm 1 lần duy nhất):** Zero-Shot `20,34%` Top-1 / `18,30%` Macro-F1 → **`66,10%` Top-1 (`156/236`)**, **`89,41%` Top-3 (`211/236`)**, **`65,36%` Macro-F1 (27 lớp)** *(Flip-TTA: `66,53%` Top-1 / `65,75%` Macro-F1; `eval_engine.py`: `63,02%` Macro-F1, `0,52 ms/ảnh` / `1.937,4 FPS`)*.
  * **Giữ kiến thức Phòng Lab (`data_v2`, đủ 38 lớp):** Valid `98,63%` Macro-F1 (`98,93%` Acc, `0,21 ms/ảnh`) | Held-Out Test `98,94%` Macro-F1 (`99,10%` Acc).
* **Xem đầy đủ các khung `📊 BẢNG ĐÁNH GIÁ CHUẨN` và bảng so sánh với Long & Khôi tại:** `notebooks/BENCHMARK_REPORT_EFFICIENTNET_B0_TRI.md`.
