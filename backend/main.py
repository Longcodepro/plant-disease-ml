"""
FastAPI Prediction Service
Tham khảo: Designing Machine Learning Systems (Chip Huyen) - Chương 7 (Model Serving)
"""

import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Thêm thư mục gốc vào PYTHONPATH để nạp modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.schemas import PredictionResponse, HealthResponse
from backend.service import ModelService

app = FastAPI(
    title="Plant Disease Classification API",
    description="REST API dự đoán bệnh trên lá cây sử dụng Deep Learning (PlantVillage Dataset)",
    version="1.0.0"
)

# CORS Middleware hỗ trợ kết nối từ Web Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo model service singleton (có thể cấu hình qua biến môi trường MODEL_PATH)
import os
model_path = os.getenv("MODEL_PATH", "models/baseline1_simple_cnn.pth")
mapping_path = os.getenv("MAPPING_PATH", "models/class_indices.json")

service = ModelService(
    model_path=model_path,
    mapping_path=mapping_path
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống phục vụ Liveness / Readiness Probes."""
    return {
        "status": "healthy" if service.model is not None else "degraded",
        "model_loaded": service.model is not None,
        "model_version": service.model_version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint nhận ảnh lá cây qua multipart/form-data và trả về kết quả dự đoán kèm độ trễ."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên bắt buộc phải là định dạng ảnh.")

    try:
        image_bytes = await file.read()
        predicted_class, confidence, top_3, latency_ms = service.predict(image_bytes)
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3": top_3,
            "inference_time_ms": latency_ms,
            "model_version": service.model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận nội bộ: {str(e)}")
