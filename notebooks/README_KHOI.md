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

### Tiến trình từng Epoch trên Kaggle (`GPU T4 x2`)

| Epoch | Thời gian | Train Acc | Lab Clean Valid Acc | PlantDoc Test Top-1 | PlantDoc Test Macro-F1 | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| **1/8** | 657.1s (~11.0p) | 93.34% | 99.40% | **65.68%** | **62.54%** | Lưu `convnext_tiny_field_v2.pth` (+34.75% Top-1 so với bản chưa domain adapt) |
| **2/8 – 8/8** | *Đang chạy* | — | — | — | — | *Sẽ cập nhật sau khi Kaggle hoàn tất* |

### Kết quả xuất ONNX INT8 & Đánh giá cuối (`plantdoc_domain_shift_v2.json`)
- *Đang chờ Kaggle chạy xong Epoch 8 và Cell 4 (xuất `convnext_tiny_field_v2_int8.onnx` + chạy `eval_domain_shift.py`).*
