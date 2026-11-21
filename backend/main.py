"""
FastAPI Prediction Service
Tham khảo: Designing Machine Learning Systems (Chip Huyen) - Chương 7 (Model Serving)
"""

import os
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Thêm thư mục gốc vào PYTHONPATH để nạp modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.schemas import BatchPredictionResponse, HealthResponse, PredictionResponse
from backend.service import ModelService

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB / ảnh
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))

app = FastAPI(
    title="Plant Disease Classification API",
    description="REST API dự đoán bệnh trên lá cây sử dụng Deep Learning (PlantVillage Dataset)",
    version="1.1.0",
)

# CORS Middleware hỗ trợ kết nối từ Web Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _default_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path
    int8_path = Path("models/convnext_tiny_khoi_int8.onnx")
    return str(int8_path) if int8_path.exists() else "models/baseline1_simple_cnn.pth"


service = ModelService(
    model_path=_default_model_path(),
    mapping_path=os.getenv("MAPPING_PATH", "models/class_indices.json"),
)


async def _read_validated_image(file: UploadFile) -> bytes:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' phải có định dạng ảnh (nhận '{content_type or 'unknown'}').",
        )
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' rỗng.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return image_bytes


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống phục vụ Liveness / Readiness Probes."""
    return {
        "status": "healthy" if service.model is not None else "degraded",
        "model_loaded": service.model is not None,
        "model_version": service.model_version,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint nhận 1 ảnh lá cây (multipart/form-data) và trả về kết quả dự đoán kèm độ trễ."""
    image_bytes = await _read_validated_image(file)
    try:
        # Chạy CPU-bound inference trong threadpool để không block async event loop
        return await run_in_threadpool(service.predict, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận nội bộ: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Endpoint nhận nhiều ảnh cùng lúc (tối đa MAX_BATCH_SIZE) và chạy 1 batch ONNX/PyTorch duy nhất."""
    if not files:
        raise HTTPException(status_code=400, detail="Cần tải lên ít nhất 1 ảnh.")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Tối đa {MAX_BATCH_SIZE} ảnh mỗi request batch (nhận {len(files)}).",
        )

    filenames = [f.filename or f"image_{i}" for i, f in enumerate(files)]
    images_bytes = [await _read_validated_image(f) for f in files]

    try:
        items, total_ms = await run_in_threadpool(service.predict_batch, images_bytes)
        results = [{"filename": name, **item} for name, item in zip(filenames, items)]
        return {
            "count": len(results),
            "total_inference_time_ms": total_ms,
            "model_version": service.model_version,
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận batch: {str(e)}")
