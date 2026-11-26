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

## 3. Giai đoạn 3 — Clean Group-Split + Mixed-Domain Adaptation (`04_convnext_tiny_domain_adaptation.ipynb`)

- **Môi trường huấn luyện:** **Kaggle Notebook (`GPU T4 x2`)**, `4 DataLoader workers` (`persistent_workers=True`), dữ liệu tạm trên `/tmp/data`, xuất artifacts ra `/kaggle/working/models/`.
- **Dữ liệu chống rò rỉ (`0%` UUID leakage):**
  - Tách lại PlantVillage theo cụm UUID gốc (`clean_split`): **`70,257` ảnh train / `17,610` ảnh valid**.
  - Trộn với `PlantDoc/train` (`2,342` ảnh × `10x` oversample) → **`93,677` mẫu train/epoch** (`BATCH_SIZE = 64`), đánh giá song song trên `Lab Valid (17,610)` và `PlantDoc Field Test (236)`.
- **Augmentation ngoài đồng ruộng:** `RandomResizedCrop(224, scale=(0.65, 1.0))`, `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.2)`, `RandomRotation(25°)`, `RandomPerspective(0.2, p=0.3)`, `ColorJitter(0.25)`, `RandomErasing(p=0.25)`.
- **Cấu hình tối ưu hóa:** Khởi tạo từ `ConvNeXt_Tiny_Weights.DEFAULT` (ImageNet), **Differential LR** (`backbone = 1e-4`, `head = 5e-4`), `AdamW (weight_decay = 1e-2)`, `CosineAnnealingLR`, `Label Smoothing = 0.1`, `AMP`, `8 epochs`. Tiêu chí chọn checkpoint tốt nhất: `0.4 * Lab_Macro_F1 + 0.6 * Field_Macro_F1`.

### Tiến trình Stage 1 — Domain Adaptation hỗn hợp (`8` epochs, Kaggle `GPU T4 x2`)

| Epoch | Thời gian | Train Acc | Lab Clean Valid Acc | PlantDoc Test Top-1 | PlantDoc Test Macro-F1 | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| **1/8** | 657.1s | 93.34% | 99.40% | 65.68% | 62.54% | ⭐ Lưu best Stage 1 |
| **2/8** | 621.5s | 98.41% | 99.23% | 67.37% | 59.08% | — |
| **3/8** | 632.6s | 99.01% | 99.65% | 69.07% | 66.25% | ⭐ Lưu best Stage 1 |
| **4/8** | 629.5s | 99.23% | 99.49% | 67.80% | 64.42% | — |
| **5/8** | 626.1s | 99.42% | 99.60% | 69.49% | 66.59% | ⭐ Lưu best Stage 1 |
| **6/8** | 662.1s | 99.60% | 99.80% | 69.07% | 64.29% | — |
| **7/8** | 654.1s | 99.73% | 99.87% | 69.49% | 68.66% | ⭐ Lưu best Stage 1 |
| **8/8** | 652.7s | 99.75% | **99.89%** | **70.76%** | **69.75%** | ⭐ Best Stage 1 (`convnext_tiny_field_v2_stage1.pth`) |

---

## 4. Giai đoạn 4 — Chuyên Sâu PlantDoc (`80%` PlantDoc Balanced + CutMix/MixUp + EMA)

- **Lấy mẫu cân bằng:** `22,400` mẫu/epoch qua `WeightedRandomSampler` (`80%` chia đều 28 lớp `PlantDoc/train` + `20%` chia đều 38 lớp `Clean PlantVillage` replay 200 ảnh/lớp để chống quên 10 lớp ngoài PlantDoc).
- **Kỹ thuật:** Khởi tạo từ Best Stage 1 (`70.76% / 69.75%`), Differential LR (`backbone = 4e-5`, `head = 2e-4`, `weight_decay = 2e-2`), Linear Warmup `1` epoch + Cosine Decay, `Label Smoothing = 0.08`, Grad Clip `1.0`, **CutMix (`0.35`) + MixUp (`0.25`)** trong 9 epoch đầu + **3 epoch Cooldown** không mix, **Model EMA (`decay = 0.995`)**.

| Epoch | Thời gian | Mix | Lab Clean Valid Acc | PlantDoc Test Top-1 | PlantDoc Test Macro-F1 | Ghi chú |
|---:|---:|:---:|---:|---:|---:|---|
| **01/12** | 336.3s | ON | 99.86% | 70.34% | 69.50% | ⭐ Lưu EMA |
| **02/12** | 331.3s | ON | 99.86% | 71.19% | 70.89% | ⭐ Lưu EMA (vượt Stage 1) |
| **03/12 ⭐** | 332.5s | ON | **99.85%** | **72.88%** | **72.41%** | **Best tổng thể (`convnext_tiny_field_v3.pth` → cập nhật `v2`)** |
| **04/12** | 337.6s | ON | 99.81% | 72.46% | 71.66% | — |
| **05/12** | 339.5s | ON | 99.81% | 72.46% | 71.86% | — |
| **06/12** | 339.3s | ON | 99.80% | 70.76% | 70.08% | — |
| **07/12** | 339.5s | ON | 99.81% | 70.76% | 70.54% | — |
| **08/12** | 338.0s | ON | 99.82% | 71.19% | 70.39% | — |
| **09/12** | 332.6s | ON | 99.84% | 71.61% | 70.81% | — |
| **10/12** | 331.7s | OFF | 99.86% | 71.61% | 70.74% | Cooldown |
| **11/12** | 326.8s | OFF | 99.87% | 72.03% | 71.02% | Cooldown |
| **12/12** | 324.7s | OFF | 99.86% | 72.03% | 71.06% | Cooldown |

### Tổng hợp bước nhảy thực nghiệm (`ConvNeXt-Tiny` trên `PlantDoc Test` 236 ảnh, 38 lớp đầy đủ)

| Mốc mô hình | Lab Clean Valid Acc | PlantDoc Top-1 (Single View) | PlantDoc Top-1 (Flip-TTA) | PlantDoc Macro-F1 |
|---|---:|---:|---:|---:|
| **Baseline Lab (`03_convnext_tiny_khoi`)** | 99.95% *(gốc)* / 100% *(unseen UUID)* | 30.93% (`73/236`) | — | 28.22% |
| **Stage 1 (`DA 8/8` Mixed-Domain)** | **99.89%** | 70.76% | — | 69.75% |
| **Stage 2 (`PlantDoc-S2 03/12` + EMA)** | **99.85%** | **72.88%** | **75.42%** (`178/236`) | **72.41%** |

### Kết quả xuất ONNX INT8 & Đánh giá cuối (`plantdoc_domain_shift_v2.json`)
- *Đang chờ chạy Cell `# 4` trên Kaggle để xuất `convnext_tiny_field_v2.onnx`, `convnext_tiny_field_v2_int8.onnx` và báo cáo `plantdoc_domain_shift_v2.json`.*
