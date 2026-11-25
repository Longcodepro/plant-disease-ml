# Bộ Quy Trình Huấn Luyện & Kiểm Thử Mô Hình `EfficientNet-B0` (Plant Disease ML)

> **Tác giả:** Võ Minh Trí (`Tri`)  
> **Nhánh Git:** `feat/efficientnet-b0-tri`  
> **Phần cứng thực nghiệm:** NVIDIA GeForce RTX 3050 Laptop GPU (4.096 MiB VRAM, TGP 40 W), SSD NVMe Micron 2400, Python 3.12.16, PyTorch `2.14.1+cu126`  
> **Tài liệu nền tảng áp dụng:** *Designing Machine Learning Systems* (Chip Huyen, Chương 3–8) & *Hands-On Machine Learning* (Aurélien Géron, Chương 13–14)

---

## 1. Mục Tiêu & Nguyên Tắc Thiết Kế

Bộ quy trình này hướng dẫn tái lập trọn vẹn **2 phiên bản mô hình** phân loại **38 tình trạng cây trồng & bệnh lý** (PlantVillage: `70.295` ảnh train / `17.572` ảnh valid / `33` ảnh test độc lập) mà **không chỉnh sửa bất kỳ mã nguồn dùng chung nào** của nhóm (`notebooks/data_pipeline.py`, `notebooks/eval_engine.py`, `backend/`, `frontend/`):

1. **Phiên bản 1 (`models/efficientnet_b0_tri.pth`):** Tối ưu chuẩn thi đấu nhóm trên cùng bộ nạp dữ liệu [data_pipeline.py](data_pipeline.py) và trọng tài chung [eval_engine.py](eval_engine.py) — đạt **`99,96%` Macro-F1** trên toàn bộ `17.572` ảnh validation.
2. **Phiên bản 2 (`models/efficientnet_b0_robust_tri.pth`):** Bổ sung pipeline tiền xử lý ảnh điện thoại (`MobileLeafPreprocessor`: **Bilateral Filter + LAB-CLAHE + Adaptive Unsharp Mask**) kết hợp tăng cường nhiễu điện thoại đồng nhất lúc huấn luyện (chống *Training-Serving Skew*) — vừa giữ **`99,91%` Macro-F1** trên ảnh sạch, vừa nâng độ chính xác trên kịch bản ảnh điện thoại kết hợp mờ/tối/nhiễu/che khuất từ **`32,17%` → `97,35%`** (giảm **`96,1%`** tổng số lỗi).

### Danh sách tệp trong nhánh `feat/efficientnet-b0-tri`

| Tệp | Vai trò |
| :--- | :--- |
| `notebooks/efficientnet_b0_tri.py` | Script huấn luyện **v1** (Transfer Learning 2 pha + AMP `fp16` + Early Stopping + chấm đối chứng Baseline 1) |
| `notebooks/test_efficientnet_b0_tri.py` | 7 unit test TDD kiểm chứng kiến trúc, đóng băng/mở khóa pha 1–2, khớp 38 nhãn, tải `weights_only=True` & notebook |
| `notebooks/03_efficientnet_b0_tri.ipynb` | Notebook tổng hợp kết quả **v1**, biểu đồ hội tụ, telemetry GPU & phân tích Top 5 lỗi |
| `notebooks/efficientnet_b0_tri_results.json` | Nhật ký thực nghiệm chi tiết **v1** (10 epochs + Baseline 1 + lỗi theo lớp) |
| `notebooks/efficientnet_b0_tri_gpu_telemetry.csv` | 1.231 mẫu `nvidia-smi` đo thực tế xuyên suốt 62,97 phút train trên RTX 3050 |
| `notebooks/mobile_preprocessing_tri.py` | Bộ tiền xử lý ảnh điện thoại `MobileLeafPreprocessor`, hàm tách `0%` rò rỉ và 7 kịch bản Stress Test |
| `notebooks/test_mobile_preprocessing_tri.py` | 8 unit test TDD kiểm chứng tiền xử lý LAB-CLAHE, làm nét, giảm nhiễu, tốc độ `< 5 ms/ảnh` & pickle đa luồng Windows |
| `notebooks/efficientnet_b0_robust_tri.py` | Script tinh chỉnh **v2 Robust** trên toàn bộ 70.295 ảnh + nhiễu điện thoại + `MobileLeafPreprocessor` |
| `notebooks/efficientnet_b0_robust_tri_results.json` | Nhật ký thực nghiệm **v2 Robust** |
| `notebooks/stress_test_comparison_tri.json` | Kết quả đối chứng Stress Test 4 cấu hình (A / B / Đối chứng / C) trên 6.108 lá sạch `0%` rò rỉ |
| `notebooks/04_mobile_preprocessing_and_stress_test_tri.ipynb` | Notebook trực quan kiểm toán rò rỉ 65,24%, demo ảnh trước/sau lọc & biểu đồ Stress Test A/B/C |
| `models/efficientnet_b0_tri.pth` | Checkpoint chuẩn thi đấu **v1** (`15,76 MB`, nạp an toàn `weights_only=True`) |
| `models/efficientnet_b0_robust_tri.pth` | Checkpoint bền vững ảnh điện thoại **v2** (`15,77 MB`, nạp an toàn `weights_only=True`) |

---

## 2. Bước 1 — Chuẩn Bị Môi Trường GPU & Dữ Liệu

### 2.1. Tạo môi trường ảo `.venv` (Python 3.12 + PyTorch CUDA)

> Không cài thư viện trực tiếp vào Python toàn cục; luôn dùng môi trường ảo `.venv` độc lập (đã có sẵn trong `.gitignore`).

**Trên Windows (PowerShell):**
```powershell
# Tạo môi trường ảo dùng Python 3.12 (tương thích tốt với wheel PyTorch CUDA)
uv venv --python 3.12 .venv

# Cài PyTorch + Torchvision có hỗ trợ GPU CUDA 12.6
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Cài các thư viện ML, xử lý ảnh (OpenCV headless), kiểm thử (pytest) và Jupyter Notebook
uv pip install --python .venv\Scripts\python.exe scikit-learn pandas matplotlib seaborn pillow opencv-python-headless pytest ipykernel nbformat nbconvert jupyter
```

**Trên Linux (Ubuntu / Linux Mint):**
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python scikit-learn pandas matplotlib seaborn pillow opencv-python-headless pytest ipykernel nbformat nbconvert jupyter
```

*Giải thích cờ lệnh nhanh:*
* `--python 3.12`: Chỉ định phiên bản Python cho môi trường ảo.
* `--index-url .../whl/cu126`: Lấy bản PyTorch biên dịch kèm thư viện GPU NVIDIA CUDA 12.6 thay vì bản chỉ chạy CPU.
* `opencv-python-headless`: Bản OpenCV nhẹ không kèm giao diện cửa sổ GUI, dùng cho `Bilateral Filter` và `LAB-CLAHE`.

**Kiểm chứng GPU nhận CUDA thành công:**
```powershell
.venv\Scripts\python.exe -c "import torch, cv2; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), 'OpenCV:', cv2.__version__)"
```

### 2.2. Chuẩn bị thư mục dữ liệu `data/`

* **Cách 1 (Tải tự động từ Google Drive của nhóm):**
  ```bash
  python scripts/download_data.py
  ```
* **Cách 2 (Đã tải sẵn file nén `.tar.gz` / `.zip` trên máy):**
  ```bash
  python scripts/download_data.py --file "/duong/dan/toi/data.tar.gz"
  ```
* **Xác nhận cấu trúc chuẩn sau khi gắn dữ liệu:**
  * `data/train/`: **38 thư mục lớp** — **70.295 ảnh**
  * `data/valid/`: **38 thư mục lớp** — **17.572 ảnh**
  * `data/test/`: **33 ảnh** kiểm thử độc lập
  * Thư mục `data/` nằm trong `.gitignore`, tuyệt đối không đẩy ảnh lên Git.

---

## 3. Bước 2 — Kiểm Thử Đơn Vị (TDD) & Chạy Thử Nhanh (Dry-Run)

Trước khi chạy huấn luyện toàn phần tốn thời gian trên GPU, luôn chạy kiểm thử đơn vị và kiểm tra mức tiêu thụ VRAM trên 2% dữ liệu:

```powershell
# 1. Chạy toàn bộ 15 unit test (kiểm tra khớp 38 lớp, đóng băng pha 1/2, pickle Windows, tốc độ tiền xử lý, tải weights_only=True)
.venv\Scripts\pytest.exe -v

# 2. Chạy thử nhanh 2% dữ liệu (rải đều 38 lớp qua np.linspace) kiểm tra VRAM < 3.5 GB
.venv\Scripts\python.exe notebooks/efficientnet_b0_tri.py --epochs_phase1 1 --epochs_phase2 1 --batch_size 32 --num_workers 2 --subset_fraction 0.02 --save_path "models/_dryrun_tmp.pth" --history_out "notebooks/_dryrun_tmp.json"
```

*Lưu ý về tối ưu phần cứng đã đo thực tế trên RTX 3050 (4 GB VRAM):*
* **AMP `fp16` vs `fp32` (`batch_size=32`):** `AMP fp16` đạt **`285,3 img/s`** với Peak VRAM **`1.461 MB`**, so với `fp32` là `161,5 img/s` và `2.801 MB` (**nhanh hơn +70,3%**, **giảm -47,8% VRAM**).
* **Chọn `batch_size=32` thay vì `64`:** `batch_size=64` đạt `302,8 img/s` (+6%) nhưng chiếm tới `2.845 MB` VRAM, dễ chạm trần 4 GB khi chạy cùng Windows.
* **Chống nghẽn DataLoader trên Windows:** Giữ nguyên dataset/transforms từ `data_pipeline.py` nhưng bật `num_workers=4`, `persistent_workers=True`, `prefetch_factor=2`, `pin_memory=True` giúp tốc độ nạp ảnh đạt **`722,2 img/s`** trên SSD NVMe.

---

## 4. Bước 3 — Huấn Luyện Phiên Bản 1 (`EfficientNet-B0 v1`)

### 4.1. Chiến lược huấn luyện 2 pha (Transfer Learning — Géron Ch. 14)

1. **Kiến trúc:** `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` (`4.056.226` tham số, `15,47 MB`), thay phần phân loại cuối bằng `Dropout(0.3) -> Linear(1280, 38)`.
2. **Hàm mất mát:** `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)` — vừa cân bằng tỷ lệ mẫu giữa 38 lớp (Chip Huyen Ch. 4), vừa giảm hiện tượng tự tin thái quá gây học vẹt.
3. **Pha 1 — Warm-up (`2` epochs):** Đóng băng toàn bộ phần thân `features`, chỉ huấn luyện `classifier` với `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR`.
4. **Pha 2 — Fine-tune (`8` epochs, `patience=3`):** Mở khóa 100% mạng với **Differential Learning Rate** (`features: 1e-4`, `classifier: 5e-4`), `CosineAnnealingLR(eta_min=1e-6)` và tự động lưu Best Checkpoint theo `val_macro_f1`.

### 4.2. Lệnh chạy huấn luyện toàn phần

```powershell
.venv\Scripts\python.exe notebooks/efficientnet_b0_tri.py `
  --epochs_phase1 2 `
  --epochs_phase2 8 `
  --batch_size 32 `
  --num_workers 4 `
  --patience 3 `
  --save_path "models/efficientnet_b0_tri.pth" `
  --history_out "notebooks/efficientnet_b0_tri_results.json"
```

*Giải thích cờ lệnh:*
* `--epochs_phase1 2 --epochs_phase2 8`: 2 vòng khởi động đầu ra + 8 vòng tinh chỉnh toàn mạng (tổng 10 epochs ≈ 21.960 bước cập nhật trên 70.295 ảnh).
* `--batch_size 32 --num_workers 4`: Kích thước lô tối ưu VRAM 4 GB + 4 luồng đọc ảnh nền.
* `--patience 3`: Dừng sớm (*Early Stopping*) nếu 3 epoch liên tiếp ở Pha 2 không vượt kỷ lục `Macro F1`.

### 4.3. Kết quả thực đo `v1` & Phát hiện khi thẩm định Baseline 1

| Chỉ số (`eval_engine.py`, RTX 3050) | Baseline 1 (`SimplePlantCNN`) | **EfficientNet-B0 v1 (`Tri`, Best Epoch 7)** |
| :--- | :---: | :---: |
| **Macro F1-Score (17.572 ảnh valid)** | `0,15%` *(lỗi cắt 1 lớp)* | **`99,96%`** |
| **Accuracy / Weighted F1** | `2,87%` / `0,16%` | **`99,96%` / `99,96%`** |
| **Lớp khó nhất trong 38 lớp** | — | **`99,67%`** (`Tomato___Target_Spot`) |
| **Tập test độc lập (`data/test/`, 33 ảnh)** | — | **`33 / 33 (100,00%)`** |
| **Độ trễ batch / đơn ảnh sync (GPU)** | `0,18 ms/ảnh` | **`0,82 ms/ảnh`** (`1.220,1 FPS`) / **`14,23 ms`** (P95 `18,03 ms`) |
| **Kích thước file `.pth`** | `1,64 MB` | **`15,76 MB`** |
| **Telemetry GPU thực đo (62,97 phút)** | — | GPU Util trung vị **`82%`** (P95 **`95%`**), Peak VRAM PyTorch **`1.469,2 MB`** (`35,9%`), Max Temp **`83 °C`**, điện năng **`36,39 Wh`** |

> **Phát hiện kiểm toán Baseline 1:** File `models/baseline1_simple_cnn.pth` cũ lưu `best_macro_f1 = 1.0`, nhưng khi chấm lại đủ 38 lớp chỉ đạt `0,15%` do `baseline1_simple_cnn.py:209-210` cắt mẫu bằng `list(range(...))` trên `ImageFolder` xếp theo tên lớp, rơi trọn vào lớp `0` (`Apple___Apple_scab`). Trong `efficientnet_b0_tri.py`, hàm `_make_even_subset_loader` khắc phục bằng `np.linspace` để luôn phủ đủ 38 lớp.

---

## 5. Bước 4 — Kiểm Toán Rò Rỉ, Tiền Xử Lý Ảnh Điện Thoại & Huấn Luyện Phiên Bản 2 (`Robust v2`)

### 5.1. Kiểm toán rò rỉ dữ liệu từ bộ Kaggle Augmented

Dùng `split_unseen_valid_indices` trong [mobile_preprocessing_tri.py](mobile_preprocessing_tri.py) truy vết UUID gốc của `17.572` ảnh trong `data/valid`:
* Trùng tên file chính xác: **`0 / 17.572`**.
* Ảnh anh em xoay/lật (`_90deg`, `_270deg`, `_new30degFlipLR`, ...) của lá đã có trong `data/train`: **`11.464` ảnh (`65,24%`)**.
* Tập **lá mới hoàn toàn (`0%` rò rỉ):** **`6.108` ảnh** — trên tập sạch này `v1` đạt **`99,90%` Accuracy / `99,52%` Macro-F1** (chỉ sai 6 ảnh).

### 5.2. Pipeline tiền xử lý ảnh điện thoại (`MobileLeafPreprocessor`)

Ảnh PlantVillage rất sạch, còn ảnh người dùng chụp bằng điện thoại thường bị mờ rung tay, tối bóng râm và nhiễu hạt cảm biến. `MobileLeafPreprocessor` (`~2,0 ms/ảnh`) xử lý theo 3 bước trên hệ màu **LAB** (chỉ can thiệp kênh sáng `L`, giữ nguyên sắc tố vết bệnh ở kênh `A, B`):
1. **Bilateral Filter** (`d=5, sigmaColor=25, sigmaSpace=25`) trên kênh `L`: dập nhiễu hạt nhưng giữ sắc biên vết bệnh.
2. **LAB-CLAHE** (`clipLimit=1.5, tileGridSize=(8, 8)`) trên kênh `L`: cứu vùng tối/chói nắng.
3. **Adaptive Unsharp Mask:** Chỉ kích hoạt làm nét (`amount=0.9`) khi phương sai Laplacian `< 120` (ảnh thực sự bị mờ); bỏ qua (`amount=0.0`) khi ảnh đã rõ nét.

### 5.3. Lệnh tinh chỉnh Phiên bản 2 (`EfficientNet-B0 Robust v2`)

Khởi tạo từ trọng số `v1`, huấn luyện trên **toàn bộ 100% `70.295` ảnh train** đi qua chuỗi mô phỏng nhiễu điện thoại (`GaussianBlur k=7, σ=0.8..3.0`, `ColorJitter`, `RandomGaussianNoisePIL σ=5..22`, `RandomErasing`) **trước** `MobileLeafPreprocessor` để đảm bảo nhất quán 100% giữa lúc học và lúc dự đoán (*Zero Training-Serving Skew*):

```powershell
.venv\Scripts\python.exe notebooks/efficientnet_b0_robust_tri.py `
  --epochs 3 `
  --batch_size 32 `
  --num_workers 4 `
  --init_checkpoint "models/efficientnet_b0_tri.pth" `
  --save_path "models/efficientnet_b0_robust_tri.pth" `
  --history_out "notebooks/efficientnet_b0_robust_tri_results.json"
```

### 5.4. Bảng đối chứng Stress Test 4 cấu hình trên 6.108 lá sạch (`0%` rò rỉ)

| Kịch bản mô phỏng ảnh điện thoại | Mốc A: `v1` gốc Acc / F1 (Lỗi) | Mốc B: `v1` + Preproc *(Skew)* Acc / F1 (Lỗi) | Đối chứng: `v2` *(tắt Preproc)* Acc / F1 (Lỗi) | **Mốc C: `v2 Robust` + `MobileLeafPreprocessor`** Acc / F1 (Lỗi) | Cải thiện Mốc C vs Mốc A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`clean`** *(Ảnh sạch)* | `99,90%` / `99,52%` (`6`) | `97,59%` / `87,52%` (`147`) | `99,62%` / `99,25%` (`23`) | **`99,85%` / `99,16%` (`9`)** | Giữ nguyên đỉnh cao |
| **`blur_mild`** *(Mờ nhẹ)* | `95,46%` / `83,98%` (`277`) | `88,59%` / `81,93%` (`697`) | `98,85%` / `98,97%` (`70`) | **`99,67%` / `96,48%` (`20`)** | **+4,21% Acc / +12,50% F1** |
| **`blur_heavy`** *(Mờ nặng)* | `55,09%` / `50,46%` (`2.743`) | `51,46%` / `50,36%` (`2.965`) | `95,94%` / `87,59%` (`248`) | **`99,35%` / `93,13%` (`40`)** | **+44,26% Acc / +42,67% F1** |
| **`harsh_lighting`** *(Tối râm)* | `98,43%` / `93,73%` (`96`) | `95,04%` / `85,43%` (`303`) | `97,68%` / `93,93%` (`142`) | **`99,59%` / `95,28%` (`25`)** | **+1,16% Acc / +1,55% F1** |
| **`sensor_noise`** *(Nhiễu hạt)* | `54,21%` / `48,18%` (`2.797`) | `13,02%` / `10,01%` (`5.313`) | `98,59%` / `88,63%` (`86`) | **`99,53%` / `92,77%` (`29`)** | **+45,32% Acc / +44,59% F1** |
| **`occlusion_22pct`** *(Che 22%)* | `98,20%` / `88,44%` (`110`) | `90,73%` / `81,99%` (`566`) | `98,00%` / `86,19%` (`122`) | **`99,08%` / `88,96%` (`56`)** | **+0,88% Acc / +0,52% F1** |
| **`combined_mobile`** *(Combo 4 nhiễu)* | `32,17%` / `27,27%` (`4.143`) | `5,53%` / `2,18%` (`5.770`) | `81,61%` / `67,31%` (`1.123`) | **`97,35%` / `84,75%` (`162`)** | **+65,18% Acc / giảm 96,1% lỗi** |

* **Kết quả tổng thể `v2 Robust` trên full `data/valid` (`17.572` ảnh):** **`99,91%` Macro-F1**, **`99,91%` Accuracy**, Latency **`1,0 ms/ảnh`** (`999,0 FPS`), Peak VRAM **`1.482,2 MB`**, kích thước **`15,77 MB`**.
* **Kết quả trên `data/test/` (`33` ảnh độc lập):** **`33 / 33 (100,00%)`**.
* **Đóng góp riêng của `MobileLeafPreprocessor`:** So với đối chứng `v2` tắt bộ lọc (`81,61%` trên `combined_mobile`), riêng bước tiền xử lý đóng góp thêm **+15,74% Accuracy** và **+17,44% Macro-F1**.

---

## 6. Bước 5 — Chạy Lại Notebook Trực Quan & Cách Nạp Mô Hình An Toàn

### 6.1. Chạy lại toàn bộ 2 Notebook báo cáo

```powershell
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/03_efficientnet_b0_tri.ipynb
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks/04_mobile_preprocessing_and_stress_test_tri.ipynb
```

### 6.2. Đoạn code mẫu nạp an toàn Checkpoint (phục vụ tích hợp FastAPI)

```python
import torch
from PIL import Image
from notebooks.efficientnet_b0_tri import build_efficientnet_b0
from notebooks.mobile_preprocessing_tri import get_mobile_eval_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Luôn nạp với weights_only=True để phòng chống lỗ hổng pickle (CWE-502)
ckpt = torch.load("models/efficientnet_b0_robust_tri.pth", map_location=device, weights_only=True)
model = build_efficientnet_b0(num_classes=ckpt["num_classes"], pretrained=False).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Với v2 Robust, nhớ dùng đồng nhất MobileLeafPreprocessor (use_preprocessor=True)
# Với v1 (models/efficientnet_b0_tri.pth), đặt use_preprocessor=False
transform = get_mobile_eval_transforms(image_size=224, use_preprocessor=True)
img_tensor = transform(Image.open("duong_dan_anh_la.jpg").convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.softmax(model(img_tensor), dim=1)
```

---

## 7. Tóm Tắt Kiểm Chứng Trước Khi Đẩy Code

Trước khi push nhánh `feat/efficientnet-b0-tri`, luôn xác nhận sạch sẽ:
1. `.venv\Scripts\pytest.exe -v` → **15/15 PASSED** (không lỗi import, không dùng `weights_only=False`).
2. `git status --short` → Không có thay đổi trên các file chung của nhóm (`data_pipeline.py`, `eval_engine.py`, `backend/`, `frontend/`) và thư mục `data/`, `.venv/` bị `.gitignore` loại bỏ hoàn toàn.
