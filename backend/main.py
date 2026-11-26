"""
FastAPI Prediction Service
Tham khảo: Designing Machine Learning Systems (Chip Huyen) - Chương 7 (Serving) & Chương 8 (Monitoring)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

# Thêm thư mục gốc vào PYTHONPATH để nạp modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.agronomy import infer_sample_metadata
from backend.schemas import (
    BatchPredictionResponse,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictWithXAIResponse,
    SamplesResponse,
)
from backend.service import ModelService

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB / ảnh
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
SAMPLES_DIR = PROJECT_ROOT / "data" / "test"
_ALLOWED_SAMPLE_EXTS = {".jpg", ".jpeg", ".png"}


def _parse_cors_config() -> tuple[List[str], bool]:
    default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
    raw = os.getenv("CORS_ORIGINS", default_origins).strip()
    if raw == "*":
        return ["*"], False
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins or "*" in origins:
        return ["*"], False
    return origins, True


_CORS_ORIGINS, _CORS_ALLOW_CREDENTIALS = _parse_cors_config()

app = FastAPI(
    title="PlantGuard AI — Plant Disease Classification API",
    description="REST API dự đoán bệnh trên lá cây (ONNX INT8 + Exact ONNX CAM XAI + Live Monitoring)",
    version="2.0.0",
)

# CORS Middleware hỗ trợ kết nối từ Web Frontend (tuân thủ đặc tả allow_credentials khi allow_origins=['*'])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _default_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path
    for candidate in ("convnext_tiny_field_v2_int8.onnx", "convnext_tiny_khoi_int8.onnx"):
        p = PROJECT_ROOT / "models" / candidate
        if p.exists():
            return str(p)
    return str(PROJECT_ROOT / "models" / "baseline1_simple_cnn.pth")


service = ModelService(
    model_path=_default_model_path(),
    mapping_path=os.getenv("MAPPING_PATH", str(PROJECT_ROOT / "models" / "class_indices.json")),
)


async def _read_validated_image(file: UploadFile) -> bytes:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' phải có định dạng ảnh (nhận '{content_type or 'unknown'}').",
        )
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' rỗng.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' vượt giới hạn {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return image_bytes


def _resolve_sample_path(filename: str) -> Path:
    if (
        not filename
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or ".." in filename
        or Path(filename).name != filename
    ):
        raise HTTPException(status_code=400, detail="Tên file mẫu không hợp lệ.")

    if Path(filename).suffix.lower() not in _ALLOWED_SAMPLE_EXTS:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh mẫu định dạng JPG hoặc PNG.")

    if not SAMPLES_DIR.exists():
        raise HTTPException(status_code=404, detail="Thư mục ảnh mẫu không tồn tại.")

    samples_root = SAMPLES_DIR.resolve()
    target = (SAMPLES_DIR / filename).resolve()
    if target.parent != samples_root:
        raise HTTPException(status_code=400, detail="Đường dẫn ảnh mẫu không hợp lệ.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu '{filename}'.")
    return target


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống phục vụ Liveness / Readiness Probes."""
    return {
        "status": "healthy" if service.model is not None else "degraded",
        "model_loaded": service.model is not None,
        "model_version": service.model_version,
    }


@app.get("/api/model-info", response_model=ModelInfoResponse)
def model_info():
    """Endpoint trả về thông tin kiến trúc, benchmark và cấu hình mô hình đang phục vụ."""
    return service.get_model_info()


@app.get("/api/samples", response_model=SamplesResponse)
def list_samples():
    """Quét thư mục `data/test/` và trả về danh sách ảnh mẫu kèm thông tin cây/bệnh suy luận."""
    if not SAMPLES_DIR.exists():
        return {"count": 0, "samples": []}

    items: List[Dict[str, Any]] = []
    for path in sorted(SAMPLES_DIR.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in _ALLOWED_SAMPLE_EXTS:
            plant, condition, is_healthy = infer_sample_metadata(path.name)
            items.append(
                {
                    "filename": path.name,
                    "plant": plant,
                    "condition": condition,
                    "is_healthy": is_healthy,
                    "url": f"/api/samples/{path.name}",
                }
            )
    return {"count": len(items), "samples": items}


@app.get("/api/samples/{filename}")
def get_sample_image(filename: str):
    """Phục vụ file ảnh mẫu từ `data/test/` với kiểm tra chống path traversal."""
    sample_path = _resolve_sample_path(filename)
    return FileResponse(path=str(sample_path))


@app.get("/metrics", response_model=MetricsResponse)
@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    """Endpoint giám sát trực tuyến (Latency p50/p95, Cache Hit Rate, Uncertain/Drift Rate, History)."""
    return service.get_metrics()


@app.post("/api/predict", response_model=PredictWithXAIResponse)
async def api_predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    include_xai: bool = Form(True),
):
    """
    Endpoint dự đoán chính cho PlantGuard AI:
    - Nhận file ảnh upload HOẶC `sample_name` (form field hoặc query param).
    - Chạy 1 pass duy nhất cho ONNX CAM XAI + trả về 3 lớp heatmap + khuyến nghị nông học tiếng Việt.
    """
    resolved_sample = sample_name or request.query_params.get("sample_name")
    if "include_xai" in request.query_params:
        resolved_include_xai = request.query_params["include_xai"].strip().lower() in ("1", "true", "yes")
    else:
        resolved_include_xai = include_xai

    if file is not None and (file.filename or not resolved_sample):
        image_bytes = await _read_validated_image(file)
    elif resolved_sample:
        sample_path = _resolve_sample_path(resolved_sample)
        image_bytes = sample_path.read_bytes()
    else:
        raise HTTPException(
            status_code=400,
            detail="Cần tải lên file ảnh ('file') hoặc cung cấp 'sample_name'.",
        )

    try:
        return await run_in_threadpool(service.predict_with_xai, image_bytes, resolved_include_xai)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận nội bộ: {str(e)}")


async def _handle_predict_batch(files: List[UploadFile]) -> Dict[str, Any]:
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
        avg_ms = round(total_ms / len(results), 2) if results else 0.0
        return {
            "count": len(results),
            "total_inference_time_ms": total_ms,
            "avg_time_per_image_ms": avg_ms,
            "model_version": service.model_version,
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận batch: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@app.post("/api/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Endpoint nhận nhiều ảnh cùng lúc (tối đa MAX_BATCH_SIZE) và chạy 1 batch ONNX/PyTorch duy nhất."""
    return await _handle_predict_batch(files)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Legacy endpoint nhận 1 ảnh lá cây và trả về dự đoán (có SHA-256 LRU cache & ngưỡng cảnh báo)."""
    image_bytes = await _read_validated_image(file)
    try:
        return await run_in_threadpool(service.predict, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận nội bộ: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)):
    """Legacy endpoint XAI trả về ảnh heatmap (Base64 PNG) khoanh vùng vết bệnh mô hình tập trung."""
    image_bytes = await _read_validated_image(file)
    try:
        return await run_in_threadpool(service.explain, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi sinh giải thích XAI: {str(e)}")
