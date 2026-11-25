# Bộ Test API bằng Postman — Plant Disease Classification

Thư mục này chứa bộ kiểm thử tự động sẵn sàng import trực tiếp vào **Postman** cho Backend REST API (`backend/main.py`).

## 1. Các tệp có sẵn

* `plant_disease_api.postman_collection.json`: Collection gồm **8 test case tự động** (có sẵn script kiểm tra `pm.test(...)`):
  1. `01. Health Check (GET /health)` — Kiểm tra trạng thái `healthy`, `model_loaded == true`, phản hồi `< 500 ms`.
  2. `02. Predict - Apple Scab` — Kiểm tra đầy đủ schema `PredictionResponse`, `top_3` sắp xếp giảm dần, nhãn đúng `Apple___Apple_scab`.
  3. `03. Predict - Tomato Early Blight` — Kiểm tra nhận diện đúng `Tomato___Early_blight`.
  4. `04. Predict - Potato Healthy` — Kiểm tra nhận diện đúng `Potato___healthy`.
  5. `05. Predict - Corn Common Rust` — Kiểm tra nhận diện đúng `Corn_(maize)___Common_rust_`.
  6. `06. Negative - Non-Image File (400)` — Gửi `postman/sample_invalid.txt`, kỳ vọng HTTP `400`.
  7. `07. Negative - Missing File Field (422)` — Gửi rỗng không có trường `file`, kỳ vọng HTTP `422`.
  8. `08. Negative - Corrupted Image File` — Gửi `postman/sample_corrupted.jpg`, kỳ vọng server không sập và trả JSON có `detail` (`400` hoặc `500`).
* `plant_disease_local.postman_environment.json`: Biến môi trường (`base_url = http://localhost:8000`, `min_confidence = 0.8`, `max_latency_ms = 2000`).
* `sample_invalid.txt` & `sample_corrupted.jpg`: File mẫu cho các ca kiểm thử lỗi.

## 2. Cách chạy trong Postman (4 bước)

1. Mở **Postman** → bấm **Import** → chọn cả 2 file:
   * `postman/plant_disease_api.postman_collection.json`
   * `postman/plant_disease_local.postman_environment.json`
2. Ở góc trên bên phải Postman, chọn Environment là **`Plant Disease API - Local`**.
3. Vào **Settings → General → Working Directory**, chọn thư mục gốc dự án `d:\plant-disease-ml` để Postman tự nhận đúng các đường dẫn ảnh (`data/test/...`, `postman/...`).
4. Khởi động Backend API rồi bấm **Run collection** trên collection `Plant Disease Classification API (FastAPI)`:
   ```powershell
   # Cài thư viện backend vào .venv (nếu chưa cài)
   uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt

   # Chạy server FastAPI
   .venv\Scripts\uvicorn.exe backend.main:app --port 8000
   ```

> **Lưu ý quan trọng về `backend/service.py`:** Hiện tại `backend/service.py` trên nhánh gốc mới nạp cứng kiến trúc `SimplePlantCNN` (`models/baseline1_simple_cnn.pth`). Do đó nếu bạn muốn dùng Postman test trực tiếp với checkpoint `models/efficientnet_b0_tri.pth` hoặc `models/efficientnet_b0_robust_tri.pth`, cần nâng cấp hàm `load_model()` trong `backend/service.py` để hỗ trợ nạp kiến trúc `EfficientNet-B0` (và bật `MobileLeafPreprocessor` cho bản Robust v2).
