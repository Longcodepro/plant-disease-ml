# BÁO CÁO DỰ ÁN MACHINE LEARNING: HỆ THỐNG NHẬN DIỆN BỆNH LÁ CÂY (PLANT DISEASE ML)
## Nghiên Cứu, Tinh Chỉnh ResNet-18 v2 & Đột Phá Thực Địa Với Pipeline 2 Tầng YOLOv8

---

* **Tác giả thực hiện:** Nguyễn Thành Long
* **Phiên bản kiến trúc chủ đạo:** ResNet-18 v2 (38 lớp chuẩn nông nghiệp) & YOLOv8n Leaf Detector
* **Nhánh phát triển Git:** `feat/resnet18-long`
* **Môi trường phần cứng:** GPU NVIDIA GeForce RTX 3050 Laptop (6GB VRAM) / OS: Linux Mint / Python 3.10 (Conda: MachineLearning)

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU KỸ THUẬT

### 1.1. Bối cảnh & Thách thức thực tế
Các mô hình nhận diện bệnh cây trồng hiện nay thường đạt độ chính xác gần như tuyệt đối ($>99\%$) trên các bộ dữ liệu phòng thí nghiệm (như **PlantVillage**). Tuy nhiên, khi mang mô hình ra đồng ruộng thực tế (bộ dữ liệu **PlantDoc**), hiệu năng sụt giảm nghiêm trọng (rơi xuống mức $\sim 38\%$). 
Hiện tượng này trong Machine Learning gọi là **Vực thẳm Miền dữ liệu (Domain Shift / Covariate Shift)**, bắt nguồn từ:
- Ánh sáng tự nhiên thay đổi (nắng gắt, bóng râm, chói sáng).
- Hậu cảnh phức tạp (đất cát, cành gai, quả chín, bàn tay nông dân).
- Tỷ lệ vật thể: Ảnh ngoài vườn thường chụp cả tán cây hoặc một cành gồm nhiều lá, trong khi mô hình phòng lab chỉ học từng phiến lá đơn lẻ cắt rời.

### 1.2. Mục tiêu dự án
1. **Chuẩn hóa đường ống dữ liệu (Data Pipeline):** Xây dựng bộ dữ liệu phân tách 38 lớp chuẩn nông nghiệp (`models/class_indices.json`), loại bỏ hoàn toàn rò rỉ dữ liệu (0% Data Leakage).
2. **Xây dựng Mô hình Baseline (Model 1):** Huấn luyện ResNet-18 v2 trên dữ liệu sạch phòng lab đạt độ tin cậy tuyệt đối.
3. **Giải pháp Không gian 2 Tầng (Two-Stage Pipeline):** Ứng dụng **YOLOv8** bóc tách phiến lá tự động kết hợp với **Chiến lược Cảnh báo Bệnh học (Strategy 2)**.
4. **Tinh chỉnh Thích nghi Thực địa (Model 2 - Fine-Tuning):** Đột phá độ chính xác thực địa bằng phương pháp trộn miền dữ liệu 2:1 (Field-to-Lab Mix), kết hợp cơ chế dừng sớm **Early Stopping** và kỹ thuật lượng tử hóa **INT8** phục vụ triển khai thực tế.

---

## 2. HỆ THỐNG NOTEBOOK THỰC NGHIỆM ĐỒNG BỘ (00 ĐẾN 07)

Toàn bộ quá trình nghiên cứu và mã nguồn được tổ chức thành chuỗi 8 Notebook tuần tự, sạch sẽ, đã thực thi sẵn đầy đủ biểu đồ trực quan:

| Thứ tự | Notebook | Nội dung chính |
| :---: | :--- | :--- |
| **00** | [00_kham_pha_va_tien_xu_ly_anh_la_cay.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/00_kham_pha_va_tien_xu_ly_anh_la_cay.ipynb) | Khám phá bản chất ma trận ảnh số, kỹ thuật Resize và trực quan hóa cơ chế nội suy Song tuyến tính (Bilinear Interpolation). |
| **01** | [01_eda_and_data_pipeline.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/01_eda_and_data_pipeline.ipynb) | Khám phá (EDA) 38 lớp bệnh PlantVillage, kiểm tra mất cân bằng dữ liệu và xây dựng PyTorch DataLoader. |
| **02** | [02_baseline1_simple_cnn.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/02_baseline1_simple_cnn.ipynb) | Xây dựng và đánh giá mạng CNN cơ bản (Simple CNN) làm mốc so sánh tối thiểu. |
| **03** | [03_train_resnet18_v2.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/03_train_resnet18_v2.ipynb) | Huấn luyện **Model 1 (ResNet-18 v2)** trên 97.391 ảnh phòng lab sạch (đạt 99.52% Acc). |
| **04** | [04_plantdoc_eda_and_yolo_extraction.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/04_plantdoc_eda_and_yolo_extraction.ipynb) | EDA thực địa PlantDoc, kiểm tra MD5 chống rò rỉ, phân chia tập 80/20 và trích xuất 13.960 phiến lá bằng YOLOv8. |
| **05** | [05_plantdoc_field_evaluation.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/05_plantdoc_field_evaluation.ipynb) | Kiểm thử thực địa Model 1: Đối chiếu giữa đưa ảnh thô (Monolithic) và Hệ thống 2 Tầng (Two-Stage YOLO). |
| **06** | [06_model_evaluation_and_analysis.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/06_model_evaluation_and_analysis.ipynb) | Phân tích sâu Ma trận nhầm lẫn (Confusion Matrix), đo lường độ trễ suy luận (Latency) và các điểm yếu ngoài trời. |
| **07** | [07_fine_tune_resnet18_early_stopping.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/07_fine_tune_resnet18_early_stopping.ipynb) | **Model 2:** Tinh chỉnh thích nghi thực địa với cơ chế 2:1 Domain Mix, Early Stopping, xuất bản ONNX & INT8. |
| **08** | [08_master_benchmark_and_system_evaluation.ipynb](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/notebooks/08_master_benchmark_and_system_evaluation.ipynb) | **Master Dashboard:** Tổng hợp và gom toàn bộ đánh giá chi tiết: thời gian chạy từng epoch, tốc độ phản hồi (Latency/FPS), kích thước tham số, hiện tượng đa bệnh và bảng điểm Production Readiness. |

---

## 3. QUÁ TRÌNH CHUẨN BỊ & XỬ LÝ DỮ LIỆU CHUYÊN SÂU

### 3.1. Dữ liệu Phòng thí nghiệm (PlantVillage v2)
- **Tổng số lượng:** 97.391 ảnh lá cây thuộc 38 lớp bệnh (14 loài cây trồng: Táo, Nho, Cà chua, Ngô, Khoai tây, Ớt chuông,...).
- **Phân chia:** Train (70%), Validation (15%), Test (15%) theo mã băm MD5, đảm bảo **0% Data Leakage**.

### 3.2. Dữ liệu Thực địa Ngoài đồng ruộng (PlantDoc)
- **Bản chất:** Ảnh tự nhiên chụp từ nhiều góc độ ngoài vườn, ánh sáng phức tạp.
- **Phân tách tập 80/20 chuẩn khoa học:**
  - *Tập ảnh nguyên tấm (Raw Photos):*
    - **Train:** 1.874 ảnh (80%)
    - **Validation:** 468 ảnh (20%)
    - **Test độc lập:** 236 ảnh
  - *Tập phiến lá bóc tách qua YOLO (Leaf Crops):*
    - **Train:** 10.959 lá cắt
    - **Validation:** 2.310 lá cắt
    - **Test độc lập:** 691 lá cắt
- **Đóng gói Bundle Zip Dereference (phục vụ Google Colab / Backup Drive):**
  - [data_plantdoc_80_20.zip](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/data_plantdoc_80_20.zip) (938.1 MB - 2.578 file ảnh thật, không dùng symlink ảo)
  - [data_plantdoc_leaves_80_20.zip](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/data_plantdoc_leaves_80_20.zip) (589.1 MB - 13.960 file phiến lá thật)

---

## 4. CÁC GIAI ĐOẠN PHÁT TRIỂN & KẾT QUẢ THỰC NGHIỆM CHI TIẾT

```
                                  TIẾN TRÌNH THỰC NGHIỆM VÀ KẾT QUẢ ĐỘT PHÁ
                                  
   +-------------------------------------------------------------------------------------------------------+
   | GIAI ĐOẠN 1: MÔ HÌNH NỀN TẢNG LAB (Model 1: ResNet-18 v2)                                             |
   |   - Huấn luyện trên PlantVillage v2 sạch (97.391 ảnh)                                                 |
   |   - Kết quả phòng lab: 99.52% Accuracy | 99.16% Macro F1                                              |
   +-------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v (Kiểm thử thực địa trên PlantDoc Test)
   +-------------------------------------------------------------------------------------------------------+
   | GIAI ĐOẠN 2: PHÁT HIỆN "VỰC THẲM MIỀN" (Domain Shift Collapse)                                       |
   |   - Đưa nguyên ảnh thô ngoài ruộng vào Model 1: Độ chính xác tụt dốc thảm hại còn 38.98%!             |
   |   - Nguyên nhân: Hậu cảnh đất cát, ánh nắng gắt, nhiều lá trên cùng một cành làm nhiễu CNN             |
   +-------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v (Đột phá kiến trúc)
   +-------------------------------------------------------------------------------------------------------+
   | GIAI ĐOẠN 3: BÓC TÁCH KHÔNG GIAN BẰNG YOLOv8 (Two-Stage Pipeline)                                     |
   |   - Tầng 1: YOLOv8n tách riêng từng phiến lá ngoài thực địa                                           |
   |   - Tầng 2: Áp dụng Chiến lược 2 (Cảnh báo bệnh học - Chỉ cần 1 lá có bệnh là cảnh báo cả cây)        |
   |   - Kết quả Model 1 tăng vọt: 38.98% -> 66.10% Top-1 (+27.12%) | 88.14% Top-3                         |
   +-------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v (Tinh chỉnh thích nghi mô hình)
   +-------------------------------------------------------------------------------------------------------+
   | GIAI ĐOẠN 4: TINH CHỈNH THÍCH NGHI 2:1 & EARLY STOPPING (Model 2: Fine-Tuned Best)                    |
   |   - Phát hiện & sửa lỗi nghiêm trọng: Lệch nhãn ImageFolder (28 lớp vs 38 lớp)                        |
   |   - Giải pháp Trộn miền 2:1: 10.959 lá ruộng (68.1%) + 5.130 ảnh mỏ neo lab (31.9%)                  |
   |   - Tốc độ học phân tầng (Backbone: 8e-5 | Head: 3e-4) + Early Stopping đỉnh cao tại Epoch 8          |
   |   - KẾT QUẢ ĐỈNH CAO THỰC ĐỊA:                                                                        |
   |       * Phiến lá test đơn lẻ (691 lá): Top-1 đạt 53.84% (+9.56%) | Macro F1 đạt 51.93% (+9.43%)        |
   |       * Ảnh thô cả cây nguyên tấm   : Top-1 đạt 54.24% (+15.26%)                                      |
   |       * HỆ THỐNG 2 TẦNG (YOLO + M2) : Top-1 đạt 75.00% (+8.90%) | Top-3 đạt 93.64% (+5.50%)!          |
   +-------------------------------------------------------------------------------------------------------+
```

### 4.1. Chẩn đoán nguyên nhân lỗi khi Fine-Tuning ban đầu & Cách khắc phục
Trong giai đoạn đầu tinh chỉnh trên tập lá cắt thực địa, mô hình gặp hiện tượng điểm số thấp do 2 nguyên nhân cốt lõi:
1. **Lỗi Lệch Chỉ số Nhãn (PyTorch `ImageFolder` Index Shift):**
   - Tập thực địa PlantDoc chỉ chứa 28 lớp bệnh (10 lớp không có mẫu ngoài tự nhiên).
   - `ImageFolder` mặc định tự động gán nhãn từ $0 \dots 27$.
   - Trong khi đó, mô hình ResNet-18 v2 gốc có 38 ngõ ra ($0 \dots 37$). Sự thiếu hụt này làm dịch chuyển toàn bộ số thứ tự (ví dụ: *Apple Cedar Rust* từ vị trí 2 bị đẩy thành vị trí 1), dẫn đến việc mô hình học sai hoàn toàn trọng số.
   - **Cách khắc phục:** Thiết kế lớp `PlantDatasetWithMapping` tùy biến, ép buộc mọi đường dẫn ảnh phải ánh xạ chính xác $100\%$ theo từ điển chuẩn 38 lớp trong `models/class_indices.json`.
2. **Hiện tượng Quên Kiến thức Thảm họa (Catastrophic Forgetting):**
   - Khi chỉ huấn luyện trên ảnh thực địa có nhãn nhiễu, mạng xóa sạch các đặc trưng vân lá chuẩn đã học được từ phòng lab.
   - **Cách khắc phục:** Áp dụng **Phương pháp 2: Trộn Dữ liệu 2:1 (Field-to-Lab Mix)**: Đưa 5.130 ảnh mỏ neo phòng lab phân bổ đều vào 38 lớp để neo giữ tri thức nền tảng.

---

## 5. BẢNG TỔNG HỢP SO SÁNH CHỈ SỐ TOÀN DIỆN (BENCHMARK SUMMARY)

Dưới đây là bảng số liệu kiểm định thực tế được đo lường đồng bộ trên cùng một tập kiểm thử độc lập:

| Môi trường Kiểm thử | Chỉ số Đánh giá | Model 1 (Lab Baseline v2) | Model 2 (Field Fine-Tuned v2) | Mức Tăng trưởng ($\Delta$) | Ý nghĩa Thực tiễn |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Phòng Lab (PlantVillage)** | Test Accuracy (%) | **99.52%** | **98.85%** | *-0.67%* | Duy trì xuất sắc tri thức chuẩn phòng thí nghiệm |
| | Test Macro F1 (%) | **99.16%** | **98.42%** | *-0.74%* | |
| **Lá cắt Thực địa (691 lá)** | Top-1 Accuracy (%) | 44.28% | **53.84%** | 🟢 **+9.56%** | Khả năng thích nghi vân lá thực tế tăng vọt |
| | Top-3 Accuracy (%) | 68.31% | **74.24%** | 🟢 **+5.93%** | |
| | Macro F1-Score (%) | 42.50% | **51.93%** | 🟢 **+9.43%** | Cải thiện mạnh trên các lớp bệnh hiếm ngoài ruộng |
| **Ảnh Toàn cảnh (236 ảnh cả cây)** | Monolithic Top-1 (%) | 38.98% | **54.24%** | 🟢 **+15.26%** | Chịu nhiễu hậu cảnh ngoài ruộng tốt hơn nhiều |
| *(Chưa qua YOLO)* | Monolithic Top-3 (%) | 63.14% | **72.46%** | 🟢 **+9.32%** | |
| **HỆ THỐNG 2 TẦNG** | **Two-Stage Top-1 (%)** | **66.10%** | **75.00%** | 🟢 **+8.90%** | **Đạt chuẩn triển khai thực tế nông nghiệp** |
| *(YOLOv8 + ResNet-18)* | **Two-Stage Top-3 (%)** | **88.14%** | **93.64%** | 🟢 **+5.50%** | **Gần như không bỏ sót bệnh của cây trồng** |

---

## 6. PHÁT HIỆN BỆNH HỌC NÔNG NGHIỆP: HIỆN TƯỢNG ĐA BỆNH (CO-INFECTION)

Khi vận hành Hệ thống 2 Tầng quét trên 236 bức ảnh thực tế, chúng ta thu được phát hiện có giá trị rất cao về mặt ứng dụng thực tiễn:
- Trong số 123 bức ảnh cây có từ 2 phiến lá trở lên, có tới **72 cây (chiếm 58.5%)** mà các lá trên cùng một cây được chẩn đoán ra **các lớp bệnh khác nhau**!
- **Thực tế:** Cây trồng ngoài ruộng thường bị đồng nhiễm (vừa bị nấm đốm lá ở tầng gốc, vừa bị rỉ sắt hoặc virus xoăn lá ở ngọn non), đồng thời luôn đan xen giữa lá khỏe và lá bệnh.
- **Ưu thế của Hệ thống 2 Tầng:** Mô hình 1 tầng truyền thống dùng Softmax buộc phải chọn 1 bệnh duy nhất và triệt tiêu các bệnh còn lại. Trong khi đó, Hệ thống 2 Tầng bóc tách độc lập từng lá, giúp người nông dân nắm bắt toàn diện mọi mầm bệnh đang xuất hiện trên cùng một luống cây.

---

## 7. ĐÓNG GÓI & TỐI ƯU HÓA TRIỂN KHAI (DEPLOYMENT ARTIFACTS)

Để sẵn sàng tích hợp vào Backend FastAPI và giao diện Streamlit / Web UI, các mô hình đã được đóng gói và đăng ký nghiêm ngặt:

1. **Trọng số PyTorch Checkpoint:**
   - File: [models/resnet18_long_v2_finetuned_best.pth](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/models/resnet18_long_v2_finetuned_best.pth)
   - Kích thước: **44.8 MB** (Lưu giữ đầy đủ trạng thái mô hình tốt nhất tại Epoch 8).
2. **Mô hình Xuất xưởng ONNX Lượng tử hóa INT8:**
   - File: [models/resnet18_long_v2_finetuned_best_int8.onnx](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/models/resnet18_long_v2_finetuned_best_int8.onnx)
   - Kích thước: **10.7 MB** (Giảm kích thước $4\times$, tối ưu tốc độ CPU latency xuống $\sim 12$ ms/ảnh mà không suy giảm độ chính xác).
3. **Mô hình Phát hiện Lá YOLOv8:**
   - File: `models/yolo/yolov8n_leaf.pt`
   - Kích thước: **6.0 MB** (Độ chính xác mAP50 đạt $85.3\%$).
4. **Sổ Đăng Ký Mô Hình (Model Registry):**
   - File: [models/model_registry.csv](file:///home/long/Documents/University/HOCTAP/HK7/Machine%20Learning%20(ML)/plant-disease-ml/models/model_registry.csv)
   - Đã cập nhật phiên bản chính thức:
     `2026-11-27 21:22:06, resnet18, long, plantdoc_leaves_v2_mix, v2_finetuned, 75.00%, 51.93%`

---

## 8. KẾT LUẬN & ĐỀ XUẤT NÂNG CAO

### 8.1. Kết luận
Dự án đã giải quyết thành công bài toán chuyển đổi mô hình từ lý thuyết phòng lab ra ứng dụng nông nghiệp thực tế:
- Đập tan hiện tượng sụt giảm độ chính xác thực địa bằng sự kết hợp giữa **Thị giác Máy tính 2 Tầng (Spatial Decoupling)** và **Kỹ thuật Trộn Miền dữ liệu 2:1**.
- Nâng độ chính xác Top-1 ngoài ruộng từ mức **38.98% lên 75.00%**, và Top-3 đạt mức gần như tuyệt đối **93.64%**.

### 8.2. Hướng mở rộng tiếp theo
1. **Lọc chất lượng lá (Quality Filtering):** Cài đặt ngưỡng lọc kích thước lá của YOLO ($>3\%$ diện tích ảnh) để loại bỏ hoàn toàn các mảnh lá nhòe ở hậu cảnh xa.
2. **Bầu chọn có trọng số (Confidence-Weighted Soft Voting):** Tổng hợp chẩn đoán cho cả cây bằng cách nhân xác suất với diện tích phiến lá thay vì chỉ dùng quy tắc cứng.
3. **Test-Time Augmentation (TTA):** Áp dụng đa biến đổi khi suy luận để tăng thêm $1-2\%$ độ chính xác khi đưa hệ thống lên ứng dụng di động cho nông dân.

---
*Báo cáo được hoàn thiện và xác thực trên toàn bộ mã nguồn của dự án tại `plant-disease-ml`.*
