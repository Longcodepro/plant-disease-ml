# 🌿 Plant Disease Classification (Phát hiện Bệnh Lá Cây)

Dự án học máy phát hiện bệnh trên lá cây nông nghiệp dựa trên tập dữ liệu PlantVillage (38 lớp cây trồng & bệnh lý).
Dự án được thiết kế theo kiến trúc 3 tầng độc lập (3-Tier Decoupled Architecture):
* **Training & Research (`notebooks/`):** Huấn luyện, thử nghiệm mô hình.
* **REST API Serving (`backend/`):** FastAPI phục vụ suy luận với cơ chế cắm rút mô hình (Plug & Play).
* **Web UI (`frontend/`):** Giao diện Streamlit cho người dùng tương tác tải ảnh và xem kết quả trực quan.

---

## 🚀 1. Hướng dẫn Tải Dữ liệu (Dataset Setup)

Repo này **không lưu trữ ảnh trực tiếp** để đảm bảo repo gọn nhẹ. Khi clone dự án về, bạn chỉ cần chạy đúng **1 lệnh duy nhất** sau:

```bash
python scripts/download_data.py
```
> Script đã được tích hợp sẵn link Google Drive của nhóm (`data.tar.gz` ~1.3GB). Nó sẽ:
> 1. Tự động tải bộ dữ liệu từ Google Drive.
> 2. Tự động giải nén và sắp xếp chuẩn vào `data/train` (70.295 ảnh) và `data/valid` (17.572 ảnh).
> 3. Tự động dọn dẹp file nén tạm để tiết kiệm dung lượng ổ cứng.

---

## 🧠 2. Cuộc thi Huấn luyện Mô hình & Tích hợp API

Mục tiêu của dự án là mọi người có thể tự do thử nghiệm các kiến trúc mô hình khác nhau (CNN, ResNet, EfficientNet, MobileNet, Vision Transformer,...):
1. Huấn luyện mô hình của bạn trong `notebooks/`.
2. Đến ngày tổng kết, mô hình nào đạt F1-Score / Accuracy và độ trễ (latency) tối ưu nhất trên tập test sẽ được chọn làm mô hình chính thức.
3. Để tích hợp mô hình thắng cuộc vào API:
   * Lưu file checkpoint trọng số vào thư mục `models/` (ví dụ: `models/best_model.pth`).
   * Khởi chạy API với biến môi trường chỉ định model:
     ```bash
     MODEL_PATH="models/best_model.pth" uvicorn backend.main:app --host 0.0.0.0 --port 8000
     ```

---

## ⚡ 3. Chạy Thử Hệ thống

### Bước 1: Khởi động Backend API
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```
* Kiểm tra API Swagger UI tại: `http://localhost:8000/docs`
* Kiểm tra trạng thái: `GET http://localhost:8000/health`

### Bước 2: Khởi động Frontend Web App
```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```
* Truy cập giao diện tại: `http://localhost:8501`
