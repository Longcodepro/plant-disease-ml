# 🤖 HƯỚNG DẪN DÀNH CHO AI ASSISTANT & THÀNH VIÊN DỰ ÁN
*(File này được thiết kế để thành viên trong nhóm đưa thẳng vào Cursor, Claude Code, Antigravity, ChatGPT, Copilot... giúp AI hiểu ngay cấu trúc dự án và bắt tay vào code mô hình)*

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU
- **Tên bài toán:** Phân loại bệnh trên lá cây nông nghiệp (Plant Disease Classification).
- **Tập dữ liệu:** PlantVillage Dataset (38 lớp cây trồng & bệnh tật).
- **Mục tiêu thành viên:** Mỗi thành viên tự do chọn và huấn luyện mô hình Machine Learning / Deep Learning (CNN, ResNet, EfficientNet, MobileNet, ConvNeXt, Vision Transformer, v.v.).
- **Đích đến chung:** Đến ngày tổng kết, mô hình đạt F1-Score / Accuracy cao nhất và có độ trễ suy luận (latency) tối ưu sẽ được chọn làm "Model Production" cắm trực tiếp vào REST API FastAPI của dự án.

---

## 2. QUY ƯỚC LÀM VIỆC (AI PHẢI TUÂN THỦ)

### 2.1. Nơi Làm Việc Của Thành Viên (`notebooks/`)
- Mọi code thử nghiệm, pipeline huấn luyện, fine-tuning mô hình **phải đặt trong thư mục `notebooks/`**.
- **Quy tắc đặt tên file tránh xung đột Git:**
  - Notebook: `notebooks/<stt>_<tên_model>_<tên_bạn>.ipynb` (Ví dụ: `notebooks/03_resnet50_nam.ipynb`, `notebooks/04_efficientnet_hoa.ipynb`).
  - File Python phụ trợ (nếu có): `notebooks/<tên_model>_<tên_bạn>.py`.

### 2.2. Vị Trí Dữ Liệu (`data/`)
- Dữ liệu đã được script chuẩn hóa sẵn theo cấu trúc ImageFolder của PyTorch:
  - `data/train/`: 38 thư mục lớp dùng để huấn luyện.
  - `data/valid/`: 38 thư mục lớp dùng để đánh giá / validation.
  - `data/test/`: 33 ảnh kiểm thử độc lập.
- **File ánh xạ nhãn chuẩn:** `models/class_indices.json` (chứa mapping từ `index` số sang `tên_lớp`). Mọi mô hình phải đồng nhất nhãn theo file này.
- **Tiền xử lý ảnh chuẩn (Preprocessing):**
  - Kích thước ảnh: `224 x 224`.
  - Chuẩn hóa màu ImageNet: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.

### 2.3. Quy Chuẩn Xuất File Model (`models/`)
Khi huấn luyện xong mô hình tốt nhất của bạn, hãy lưu checkpoint PyTorch vào thư mục `models/`:
- **Quy ước đặt tên checkpoint:** `models/<tên_model>_<tên_bạn>.pth` (Ví dụ: `models/resnet50_nam.pth`).
- **Nội dung dictionary lưu trong file `.pth`:**
  ```python
  import torch

  torch.save({
      "model_name": "resnet50",
      "author": "ten_thanh_vien",
      "num_classes": 38,
      "val_accuracy": val_acc,
      "val_f1_macro": val_f1,
      "model_state_dict": model.state_dict()
  }, "models/resnet50_nam.pth")
  ```

---

## 3. CÁC MODULE DÙNG CHUNG ĐÃ ĐƯỢC CHUẨN HÓA SẴN (BẮT BUỘC SỬ DỤNG)

Để đảm bảo tính công bằng tuyệt đối khi so sánh mô hình giữa các thành viên:

### 3.1. Dùng chung Data Pipeline (`notebooks/data_pipeline.py`)
Mọi thành viên **không tự viết lại hàm load dữ liệu**, hãy import trực tiếp:
```python
from notebooks.data_pipeline import build_dataloaders

train_loader, valid_loader, idx_to_class, class_weights = build_dataloaders(
    data_dir="data",
    batch_size=64
)
```

### 3.2. Dùng chung Tiêu Chí Đánh Giá (`notebooks/eval_engine.py`)
Sau khi huấn luyện xong mỗi epoch hoặc đánh giá mô hình cuối cùng, hãy gọi hàm đánh giá chuẩn dùng chung:
```python
from notebooks.eval_engine import evaluate_model, print_evaluation_summary

# Đánh giá toàn diện trên tập Validation:
metrics = evaluate_model(model, valid_loader, device=device)

# In bảng tổng kết đẹp mắt:
print_evaluation_summary(metrics, model_name="ResNet50_Nam")
```
**Các tiêu chí chuẩn được đo tự động:**
* **`macro_f1` (Tiêu chí quyết định thắng cuộc):** Trung bình F1 của 38 lớp, đảm bảo không bị bias vào lớp nhiều ảnh.
* **`accuracy`:** Độ chính xác tổng quát.
* **`latency_per_image_ms`:** Độ trễ khi dự đoán 1 bức ảnh (mili-giây), tiêu chí quan trọng để đưa vào API.
* **`model_size_mb`:** Dung lượng mô hình khi lưu trữ.
```text
plant-disease-ml/
├── data/                 # Chứa dữ liệu train/valid/test (bị ignore trên Git)
├── notebooks/            # 👈 NƠI CÁC BẠN CODE & TRAIN MÔ HÌNH TẠI ĐÂY
├── models/               # Nơi lưu checkpoint .pth và file nhãn class_indices.json
├── backend/              # FastAPI Serving (Đã viết sẵn, tự động nạp model qua biến môi trường)
├── frontend/             # Streamlit Web UI tương tác demo
├── scripts/
│   └── download_data.py  # Script tải và tự động bung dataset vào data/
└── README.md             # Hướng dẫn chung
```

---

## 4. LỆNH NHANH (QUICKSTART CHO THÀNH VIÊN)

1. **Kéo dữ liệu về máy (Chỉ 1 lệnh duy nhất):**
   ```bash
   python scripts/download_data.py
   ```
   *(Link Google Drive đã được gắn sẵn trong script, tự động tải và bung vào `data/train` & `data/valid`)*
2. **Cài đặt thư viện:**
   ```bash
   pip install -r backend/requirements.txt
   ```
3. **Chạy thử mô hình của bạn trên Backend API (Test cắm rút):**
   ```bash
   MODEL_PATH="models/<tên_model_của_bạn>.pth" uvicorn backend.main:app --reload --port 8000
   ```
4. **Mở giao diện Web test thử ảnh thực tế:**
   ```bash
   streamlit run frontend/app.py
   ```
