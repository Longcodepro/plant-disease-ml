# Báo Cáo Đánh Giá Chuẩn — `EfficientNet-B0` (PlantVillage `data_v2` & Fine-Tune Gộp `data_plantdoc` 80/20)

> **Tác giả:** Võ Minh Trí (`Tri`)  
> **Nhánh Git:** `feat/efficientnet-b0-tri`  
> **Phần cứng đo thực tế:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM, CUDA 12.6), PyTorch `2.14.1+cu126`  
> **Thước đo chuẩn hóa:** Chấm thống nhất qua `notebooks/eval_engine.py` dùng chung của nhóm (kèm thêm chỉ số 27/28 lớp chuẩn `scripts/eval_domain_shift.py` của Khôi trên tập PlantDoc)  
> **Quy trình & Kiểm định dữ liệu (đồng bộ 100% với nhóm):**
> - **Bước 1 — PlantVillage `data_v2` (`duydangkaggle/plantvillage-leaf-grouped-split`):** `38,039` train / `8,147` valid / `8,098` held-out test (`38/38` lớp, chia theo nhóm lá gốc, **0%** trùng tên/UUID/SHA-256).
> - **Bước 2 — Fine-tune gộp chung `data_plantdoc_80_20` vào `data_v2`:** Giữ nguyên bộ chia chuẩn của nhóm (`train: 1,871` ảnh / `28` lớp, `valid: 468` ảnh / `28` lớp, `test: 236` ảnh / `27` lớp; `0` ảnh hỏng; **`0` trùng SHA-256 giữa `train ↔ valid`**; ghi nhận minh bạch có `9` cặp trùng SHA-256 `test ↔ train` và `2` cặp `test ↔ valid` có sẵn từ repo gốc `PlantDoc-Dataset`). Gộp toàn bộ `38,039 + 1,871 = 39,910` ảnh train, chọn `Best EMA` **chỉ trên `PlantDoc Valid` (`468` ảnh)** + giữ `Lab Valid >= 98.5%`, và **khóa kín `PlantDoc Test` (`236` ảnh)** để chấm 1 lần duy nhất ở cuối.

---

## 1. Bước 1 — Bảng Đánh Giá Chuẩn Trên PlantVillage `data_v2` (38 lớp, Phòng Lab)

### 1.1. `EfficientNet-B0 v2` (`models/efficientnet_b0_tri.pth` — Best Epoch 13/15)

```text
=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_V2
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 99.59%
 📈 Accuracy                         : 99.68%
 ⚖️  Weighted F1-Score               : 99.68%
 🔍 Macro Precision / Recall         : 99.55% / 99.64%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.62 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 1625.6 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================


=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_V2_TEST_SET
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 99.35%
 📈 Accuracy                         : 99.62%
 ⚖️  Weighted F1-Score               : 99.62%
 🔍 Macro Precision / Recall         : 99.14% / 99.58%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.30 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 3342.7 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================
```

*Ghi chú:* Tập Valid (`8,147` ảnh) chỉ sai `26` ảnh; tập Held-Out Test chấm 1 lần (`8,098` ảnh) chỉ sai `31` ảnh. Đơn ảnh đồng bộ GPU (`batch_size=1`, 50 mẫu): trung bình `21.46 ms`, P95 `26.03 ms`. File `.pth`: `15.76 MB`.

---

### 1.2. `EfficientNet-B0 Robust v2` (`models/efficientnet_b0_robust_tri.pth` — Chống nhiễu điện thoại, Best Epoch 2/4)

```text
=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_ROBUST_V2
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 99.39%
 📈 Accuracy                         : 99.47%
 ⚖️  Weighted F1-Score               : 99.47%
 🔍 Macro Precision / Recall         : 99.36% / 99.42%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.62 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 1601.3 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================


=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_ROBUST_V2_TEST_SET
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 99.12%
 📈 Accuracy                         : 99.37%
 ⚖️  Weighted F1-Score               : 99.37%
 🔍 Macro Precision / Recall         : 99.02% / 99.22%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.30 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 3314.9 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================
```

*Ghi chú:* Kết hợp `MobileLeafPreprocessor` (LAB-CLAHE + Bilateral Filter + Adaptive Unsharp Mask, ~2 ms/ảnh) đẩy `combined_mobile` (combo 4 nhiễu điện thoại trên `8,147` ảnh) từ **`27.92%` → `93.94%` Macro-F1** (`+66.02%`, giảm `93.3%` lỗi). Held-Out Test (`8,098` ảnh) sai `51` ảnh; đơn ảnh GPU avg `21.05 ms` (P95 `25.80 ms`). File `.pth`: `15.77 MB`.

---

## 2. Bước 2 — Fine-Tuning Gộp Chung `data_v2` + `data_plantdoc` 80/20 (39,910 ảnh train)

Mô hình **`EfficientNet-B0 PlantDoc`** (`models/efficientnet_b0_plantdoc_tri.pth`) được fine-tune từ `models/efficientnet_b0_tri.pth` trên tập gộp `38,039` ảnh `data_v2/train` + `1,871` ảnh `data_plantdoc/train` = **`39,910` ảnh** (14 epochs, `31.43 phút`, Peak VRAM `1,499.4 MB`, Best EMA Epoch **`12/14`** chọn thuần trên `PlantDoc Valid`):
- **Stage 1 (Ep 1–7):** Cầu nối gộp (`55%` PlantDoc / `45%` `data_v2`, `min_pd_count=15`, CutMix `0.30` + MixUp `0.20` 5 epoch đầu + 2 epoch Cooldown).
- **Stage 2 (Ep 8–14):** Tinh chỉnh ngoài trời (`75%` PlantDoc / `25%` `data_v2` replay 38 lớp + Model EMA `decay=0.995` + 3 epoch Cooldown).
- **Khóa kín `PlantDoc Test` (`236` ảnh):** Không sử dụng để chọn epoch hay điều chỉnh tham số; chỉ mở chấm **1 lần duy nhất** ở bước cuối.

### 2.1. Trước khi Fine-Tune (Zero-Shot từ `v2` phòng lab → `PlantDoc Test`)

```text
=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_PLANTDOC_ZERO_SHOT
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 15.44%
 📈 Accuracy                         : 20.34%
 ⚖️  Weighted F1-Score               : 18.83%
 🔍 Macro Precision / Recall         : 22.82% / 16.94%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.39 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 2590.2 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================
```

*(Trước khi fine-tune: `PlantDoc Valid` đạt Top-1 `17.52%`, Top-3 `35.68%`, Macro-F1 28 lớp `16.43%`; `PlantDoc Test` đạt Top-1 `20.34%` (`48/236`), Top-3 `38.98%`, Macro-F1 27 lớp `18.30%`.)*

---

### 2.2. Sau Fine-Tune Gộp Chung — Chấm Trên `PlantDoc Valid` (468 ảnh, 28 lớp) & `PlantDoc Test` (236 ảnh, 27 lớp, Chấm 1 Lần)

```text
=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_PLANTDOC_VALID
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 70.68%
 📈 Accuracy                         : 70.94%
 ⚖️  Weighted F1-Score               : 70.86%
 🔍 Macro Precision / Recall         : 72.60% / 70.76%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.45 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 2202.3 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================


=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_PLANTDOC_TEST_SET
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 63.02%
 📈 Accuracy                         : 66.10%
 ⚖️  Weighted F1-Score               : 66.08%
 🔍 Macro Precision / Recall         : 66.20% / 63.72%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.52 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 1937.4 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================
```

> **Chỉ số quy đổi theo đúng chuẩn 27/28 lớp có mặt (`scripts/eval_domain_shift.py` của Khôi):**
> - **Trên `PlantDoc Valid` (`468` ảnh, 28 lớp):** Top-1 **`70.94%`** (`332/468`, **+53.42%** vs Zero-Shot), Top-3 **`89.10%`**, Macro-F1 (28 lớp) **`70.68%`** (gấp **`4.30×`** Zero-Shot `16.43%`!).
> - **Trên `PlantDoc Test` (`236` ảnh, 27 lớp, chấm 1 lần không rò rỉ):**
>   - **Single-Pass:** Top-1 **`66.10%`** (`156/236`, **+45.76%** / `3.25×` vs Zero-Shot), Top-3 **`89.41%`** (`211/236`, **+50.43%**), Macro-F1 (27 lớp) **`65.36%`** (Precision / Recall: `68.65% / 66.08%`).
>   - **Flip-TTA (lật ngang):** Top-1 **`66.53%`** (`157/236`), Top-3 **`88.98%`**, Macro-F1 (27 lớp) **`65.75%`**.
> - *(Giải thích chênh lệch `63.02%` vs `65.36%` trên Test: hàm `eval_engine.py` chia Macro-F1 trên hợp `set(y_true) | set(y_pred)`, nên một vài ảnh đoán sang 11 lớp ngoài PlantDoc làm mẫu số chia tăng từ 27 lên 28.)*

---

### 2.3. Kiểm Chứng Giữ Kiến Thức Phòng Lab (`PlantDoc Fine-Tuned Model` chấm lại trên `data_v2` 38 lớp)

```text
=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_PLANTDOC_ON_LAB_VALID
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 98.63%
 📈 Accuracy                         : 98.93%
 ⚖️  Weighted F1-Score               : 98.94%
 🔍 Macro Precision / Recall         : 98.46% / 98.84%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 0.21 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 4739.0 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================


=======================================================
 📊 BẢNG ĐÁNH GIÁ CHUẨN: EFFICIENTNET_B0_TRI_PLANTDOC_ON_LAB_TEST_SET
=======================================================
 🎯 Macro F1-Score (Tiêu chí chính) : 98.94%
 📈 Accuracy                         : 99.10%
 ⚖️  Weighted F1-Score               : 99.10%
 🔍 Macro Precision / Recall         : 98.75% / 99.15%
-------------------------------------------------------
 ⚡ Độ trễ suy luận (Latency)        : 2.49 ms/ảnh
 🚀 Tốc độ xử lý (Throughput)        : 402.4 khung hình/giây (FPS)
 📦 Dung lượng mô hình               : 15.47 MB (4,056,226 tham số)
=======================================================
```

*(Nhờ fine-tune gộp chung với `data_v2`, mô hình vẫn giữ vững **`99.10%` Accuracy / `98.94%` Macro-F1** trên toàn bộ `8,098` ảnh Held-Out Test đủ 38 lớp phòng lab, không hề bị quên thảm họa các lớp không có trong PlantDoc.)*

---

## 3. So Sánh Tổng Quan Với Các Mô Hình Trong Nhóm

| Mô hình (Tác giả) | Tham số / Dung lượng FP32 | Lab `data_v2` Valid Macro-F1 / Acc | Lab `data_v2` Test Macro-F1 / Acc | PlantDoc Valid (468) Top-1 / Macro-F1 | PlantDoc Test (236, 27 lớp) Top-1 / Top-3 / Macro-F1 | Ghi chú chuẩn đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RESNET18_LONG_V2`** *(Long)* | `11,196,006` (`42.71 MB`) | `99.16%` / `99.52%` | `98.95%` / `99.35%` | — | — | `0.23` / `0.16 ms/ảnh` |
| **`EFFICIENTNET_B0_TRI_V2`** *(Trí)* | **`4,056,226` (`15.47 MB`)** | **`99.59%` / `99.68%`** | **`99.35%` / `99.62%`** | `17.52%` / `16.43%` *(Zero-Shot)* | `20.34%` / `38.98%` / `18.30%` *(Zero-Shot)* | Nhẹ hơn ResNet-18 **`2.76×`**, vượt Long cả Valid (`+0.43%`) & Test (`+0.40%`) |
| **`EFFICIENTNET_B0_TRI_ROBUST_V2`** *(Trí)* | **`4,056,226` (`15.47 MB`)** | `99.39%` / `99.47%` | `99.12%` / `99.37%` | — *(Stress 4 nhiễu: `93.94%` F1)* | — | Có `MobileLeafPreprocessor` chống mờ/nhiễu/ánh sáng |
| **`EFFICIENTNET_B0_TRI_PLANTDOC`** *(Trí)* | **`4,056,226` (`15.47 MB`)** | `98.63%` / `98.93%` | `98.94%` / `99.10%` | **`70.94%` / `70.68%`** | **`66.10%` / `89.41%` / `65.36%`** *(TTA: `66.53%` / `65.75%`; eval_engine: `63.02%`)* | **Đúng chuẩn 80/20**: chọn Best EMA thuần trên Valid (`468`), khóa kín Test (`236`) chấm 1 lần; nhẹ hơn ConvNeXt **`6.87×`** |
| **`CONVNEXT_TINY_FIELD_V2`** *(Khôi)* | `27,849,350` (`106.3 MB` FP32 / `32.3 MB` INT8) | `99.70%` Acc *(clean split)* | — | — | `73.31%` / `92.37%` / `71.94%` *(train trên 100% PlantDoc train `2,342`, chọn Best trên Test)* | Khác giao thức (dùng 100% `2,342` ảnh train thay vì 80% `1,871` ảnh và chọn Best EMA trên tập Test) |

---

## 4. Phân Tích Lỗi Trên `PlantDoc Valid` (468) & `PlantDoc Test` (236)

### Top 5 cặp nhầm lẫn trên `PlantDoc Valid` (`468` ảnh)
| Hạng | Nhãn thật | Dự đoán nhầm | Số ảnh |
| :---: | :--- | :--- | :---: |
| 1 | `Potato___Early_blight` | `Potato___Late_blight` | `6` |
| 2 | `Tomato___Bacterial_spot` | `Tomato___Septoria_leaf_spot` | `6` |
| 3 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn_(maize)___Northern_Leaf_Blight` | `5` |
| 4 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `4` |
| 5 | `Potato___Early_blight` | `Tomato___Early_blight` | `4` |

### Top 5 cặp nhầm lẫn trên `PlantDoc Test` (`236` ảnh, chấm 1 lần)
| Hạng | Nhãn thật | Dự đoán nhầm | Số ảnh |
| :---: | :--- | :--- | :---: |
| 1 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `7` |
| 2 | `Tomato___Bacterial_spot` | `Tomato___Septoria_leaf_spot` | `4` |
| 3 | `Tomato___Septoria_leaf_spot` | `Tomato___Bacterial_spot` | `4` |
| 4 | `Blueberry___healthy` | `Apple___healthy` | `3` |
| 5 | `Potato___Early_blight` | `Potato___Late_blight` | `3` |

### Lớp tốt nhất & khó nhất trên `PlantDoc Test` (`236` ảnh)
- **Tốt nhất (F1 ≥ 90%):** `Strawberry___healthy` (`94.12%`, 8/8), `Raspberry___healthy` (`93.33%`, 7/7), `Grape___healthy` (`92.31%`, 12/12), `Squash___Powdery_mildew` (`90.91%`, 5/6), `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (`90.91%`, 5/6).
- **Khó nhất (F1 ≤ 35%):** `Tomato___Leaf_Mold` (`15.38%`, 1/6), `Tomato___Bacterial_spot` (`23.53%`, 2/9), `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` (`26.67%`, 2/4), `Tomato___Tomato_mosaic_virus` (`33.33%`, 2/10), `Potato___Early_blight` (`35.29%`, 3/8).

---

## 5. Tài Nguyên Kiểm Chứng & Tái Lập

- **Script fine-tune gộp chung:** `notebooks/efficientnet_b0_plantdoc_tri.py`
- **Unit tests TDD (23/23 PASSED):** `notebooks/test_efficientnet_b0_plantdoc_tri.py` (8), `notebooks/test_efficientnet_b0_tri.py` (7), `notebooks/test_mobile_preprocessing_tri.py` (8)
- **Log JSON đầy đủ:** `notebooks/efficientnet_b0_plantdoc_tri_results.json`, `notebooks/efficientnet_b0_tri_results.json`, `notebooks/efficientnet_b0_robust_tri_results.json`, `notebooks/data_v2_final_test_results_tri.json`
- **Checkpoints (`weights_only=True`):**
  - `models/efficientnet_b0_tri.pth` (`15.76 MB`)
  - `models/efficientnet_b0_robust_tri.pth` (`15.77 MB`)
  - `models/efficientnet_b0_plantdoc_tri.pth` (`15.77 MB`)
