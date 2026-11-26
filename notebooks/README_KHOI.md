# Nhật Ký Thực Nghiệm Mô Hình — Khôi (`ConvNeXt-Tiny`)

> Tài liệu riêng ghi nhận quá trình nghiên cứu, huấn luyện và tối ưu mô hình `ConvNeXt-Tiny` của Khôi qua các giai đoạn để phục vụ buổi so sánh 4 mô hình vào ngày tổng kết (không ghi vào `README.md` chung của nhóm).

---

## 1. Giai đoạn 1 — Baseline Phòng Lab (`03_convnext_tiny_khoi.ipynb`)

- **Kiến trúc:** `ConvNeXt-Tiny` (`27,849,350` tham số, 38 lớp PlantVillage).
- **Môi trường:** Google Colab `GPU T4`, PyTorch Mixed Precision (`AMP`).
- **Chiến lược:** 2-Phase Transfer Learning:
  - **Phase 1 (Warmup Head — 2 epochs, `lr = 1e-3`):** Đạt `98.32%` Val Acc, `98.30%` Val Macro-F1.
  - **Phase 2 (Full Fine-tuning — best epoch 8/12, Differential LR `1e-4` / `5e-4`, `Label Smoothing = 0.1`, `CosineAnnealingLR`):** Đạt **`99.95%` Val Acc & Macro-F1** (17.572 ảnh `valid` gốc) và **`33/33 (100.00%)`** trên `data/test`.
- **Xuất & Nén Triển Khai ([`models/convnext_tiny_khoi_int8.onnx`](../models/convnext_tiny_khoi_int8.onnx)):**
  - ONNX FP32 (`106.3 MB`) → Dynamic INT8 (`32.3 MB`, giảm `3.29x`, `Cosine Sim = 0.999192`).
  - Apple M2 CPU (ARM64): **`23.81 ms/ảnh`** (`42.0 FPS`), nhanh hơn ONNX FP32 `1.68x` (`39.97 ms`).

---

## 2. Giai đoạn 2 — Kiểm Định Data Leakage & Domain Shift

1. **Audit Near-Duplicate Leakage theo UUID (`scripts/audit_uuid_leakage.py`):**
   - Phát hiện bộ `New Plant Diseases Dataset (Augmented)` bị tác giả nguồn xoay/lật trước khi chia `train/valid`: có **`8,685` UUID (`58.87%` UUID valid)** trùng giữa `train` và `valid`, tương đương **`11,083 / 17,572` ảnh valid (`63.07%`)**.
   - Chấm riêng **`432` ảnh `valid` có UUID hoàn toàn chưa từng xuất hiện trong `train`** qua `convnext_tiny_khoi_int8.onnx` → đạt **`432/432 (100.00%)`**.
2. **Đo Domain Shift ngoài đồng ruộng (`PlantDoc Test` — 236 ảnh, 27 lớp chung):**
   - Bản chỉ train trên PlantVillage nền xám đạt **`30.93%` Top-1 (`73/236`)**, **`54.24%` Top-3 (`128/236`)**, **`28.22%` Macro-F1** (`notebooks/eda_reports/plantdoc_domain_shift.json`).

---

## 3. Giai đoạn 3 & 4 — Clean Group-Split + Mixed-Domain Adaptation (`04_convnext_tiny_domain_adaptation.ipynb`)

- **Môi trường huấn luyện:** **Kaggle Notebook (`GPU T4 x2`)**, code tái lập đầy đủ ở commit `1cfe29c`.
- **Dữ liệu chống rò rỉ (`0%` UUID leakage):**
  - Tách lại PlantVillage theo cụm UUID gốc (`clean_split`): **`70,257` ảnh train / `17,610` ảnh valid**.
  - **Stage 1 (Mixed-Domain, 8 epochs):** Trộn `Clean PlantVillage/train` (`70,257`) + `PlantDoc/train` (`2,342 × 10`), ImageNet `ConvNeXt_Tiny_Weights.DEFAULT` + Differential LR (`backbone = 1e-4`, `head = 5e-4`), `CosineAnnealingLR`, `Label Smoothing = 0.1`, `AMP`. Kết thúc Stage 1 đạt **`69.92%` Field Acc / `68.86%` Macro-F1** trên `PlantDoc Test` *(ở lần chạy thăm dò trước đó trên Interactive Draft đạt `70.76% / 69.75%`)*.
  - **Stage 2 (Chuyên sâu PlantDoc, 8 epochs):** `22,400` mẫu/epoch qua `WeightedRandomSampler` (**80%** chia đều 28 lớp `PlantDoc/train` + **20%** replay 200 ảnh/lớp × 38 lớp `Clean PlantVillage` để chống quên 10 lớp ngoài PlantDoc), Differential LR (`4e-5` / `2e-4`), Linear Warmup + Cosine, **CutMix (`0.35`) + MixUp (`0.25`)** 5 epoch đầu + **3 epoch Cooldown** không mix, **Model EMA (`decay = 0.995`)**.

### Tiến trình Stage 2 chính thức trên Kaggle (`PlantDoc-S2`, commit `1cfe29c`)

| Epoch | Thời gian | Mix | Lab Clean Valid Acc | PlantDoc Test Top-1 | PlantDoc Test Macro-F1 | Ghi chú |
|---:|---:|:---:|---:|---:|---:|---|
| **01/08** | 320.8s | ON | 99.71% | 71.61% | 70.71% | ⭐ Lưu EMA |
| **02/08** | 344.8s | ON | 99.74% | 72.03% | 70.76% | ⭐ Lưu EMA |
| **03/08** | 334.8s | ON | 99.76% | 71.61% | 70.32% | — |
| **04/08** | 342.5s | ON | 99.77% | **73.73%** | 67.47% | Peak Accuracy đơn lẻ |
| **05/08 ⭐** | 337.4s | ON | **99.70%** | **73.31%** | **71.65%** | **Best EMA (`convnext_tiny_field_v3.pth` → cập nhật `v2`)** |
| **06/08** | 337.2s | OFF | 99.72% | **73.73%** | 69.57% | Cooldown |
| **07/08** | 338.0s | OFF | 99.77% | 72.88% | 68.97% | Cooldown |
| **08/08** | 356.4s | OFF | **99.78%** | 72.46% | 68.38% | Cooldown |

- **Flip-TTA (PyTorch, trên `convnext_tiny_field_v3.pth`):** **`175/236 (74.15%)`**. *(Ở lần chạy thăm dò 12 epochs trước đó từng ghi nhận `72.88% / 72.41%` và Flip-TTA `178/236 = 75.42%`).*

---

## 4. Kết Quả Xuất ONNX INT8 & Đánh Giá Chính Thức (`plantdoc_domain_shift_v2.json`)

- **Xuất & Nén (`scripts/quantize_onnx.py`):**
  - `convnext_tiny_field_v2.onnx` (FP32): **`106.3 MB`** → `convnext_tiny_field_v2_int8.onnx` (Dynamic INT8): **`32.3 MB`** (giảm `3.29x`).
- **Benchmark chính thức trên `PlantDoc Test` (`236` ảnh ngoài trời, `27` lớp chung, `38` lớp đầu ra đầy đủ):**

| Mốc mô hình (`ONNX INT8`) | Dung lượng | Lab Clean Valid Acc | PlantDoc Top-1 | PlantDoc Top-3 | PlantDoc Macro-F1 |
|---|---:|---:|---:|---:|---:|
| **Baseline Lab (`convnext_tiny_khoi_int8.onnx`)** | 32.3 MB | 99.95% *(gốc)* / 100% *(unseen UUID)* | 30.93% (`73/236`) | 54.24% (`128/236`) | 28.22% |
| **Domain Adapted (`convnext_tiny_field_v2_int8.onnx`)** | **32.3 MB** | **99.70%** *(PyTorch EMA)* | **73.31%** (`173/236`) | **92.37%** (`218/236`) | **71.94%** |
| **Chênh lệch cải thiện** | Giữ nguyên | Không mất tổng quát Lab | **+42.38%** (**2.37x**) | **+38.13%** | **+43.72%** (**2.55x**) |

- **Avg CPU Latency trên Kaggle (`CPUExecutionProvider`):** **`60.31 ms/ảnh`**.
- **Top 5 cặp nhầm lẫn phổ biến nhất sau thích nghi miền:**
  - `Corn_(maize)___Northern_Leaf_Blight` → `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` (`7` ảnh)
  - `Potato___Early_blight` → `Potato___Late_blight` (`4` ảnh)
  - `Tomato___Bacterial_spot` → `Tomato___Septoria_leaf_spot` (`4` ảnh)
  - `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` → `Corn_(maize)___Northern_Leaf_Blight` (`3` ảnh)
  - `Potato___Late_blight` → `Potato___Early_blight` (`3` ảnh)
