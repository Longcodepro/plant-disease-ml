"""
Pydantic Schemas for Prediction API
Tham khảo: Designing Machine Learning Systems (Chip Huyen) - Chương 7, 8 & 9
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str = Field(..., description="Tên lớp bệnh hoặc tình trạng cây")
    confidence: float = Field(..., description="Độ tin cậy xác suất (0.0 - 1.0)")


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Nhãn có xác suất cao nhất")
    confidence: float = Field(..., description="Độ tin cậy của nhãn dự đoán cao nhất")
    top_3: List[PredictionItem] = Field(default_factory=list, description="Top 3 dự đoán có xác suất cao nhất")
    is_uncertain: bool = Field(False, description="True khi độ tin cậy thấp hơn ngưỡng cảnh báo")
    warning: Optional[str] = Field(None, description="Cảnh báo khi ảnh mờ hoặc có thể không phải lá cây thuộc 38 lớp")
    cache_hit: bool = Field(False, description="True nếu kết quả được trả tức thì từ SHA-256 LRU cache")
    inference_time_ms: float = Field(..., description="Thời gian thực thi suy luận (ms)")
    model_version: str = Field(..., description="Phiên bản hoặc tên mô hình đang phục vụ")


class BatchPredictionItem(BaseModel):
    filename: str = Field(..., description="Tên file ảnh trong batch")
    predicted_class: str
    confidence: float
    top_3: List[PredictionItem] = Field(default_factory=list)
    is_uncertain: bool = False
    warning: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    count: int = Field(..., description="Số lượng ảnh đã dự đoán trong batch")
    total_inference_time_ms: float = Field(..., description="Tổng thời gian suy luận cho cả batch (ms)")
    model_version: str
    results: List[BatchPredictionItem] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: List[PredictionItem] = Field(default_factory=list)
    is_uncertain: bool = False
    warning: Optional[str] = None
    method: str = Field(..., description="Phương pháp giải thích XAI (Occlusion Sensitivity Heatmap)")
    heatmap_base64: str = Field(..., description="Ảnh PNG heatmap chồng lên lá cây mã hóa Base64")
    explain_time_ms: float = Field(..., description="Tổng thời gian sinh giải thích heatmap (ms)")
    model_version: str


class LatencyStats(BaseModel):
    avg: float
    p50: float
    p95: float
    min: float
    max: float


class MetricsResponse(BaseModel):
    uptime_seconds: float
    model_version: str
    backend_type: str
    total_requests: int
    total_images_predicted: int
    cache_hits: int
    cache_hit_rate: float
    uncertain_count: int
    uncertain_rate: float
    latency_ms: LatencyStats
    top_predicted_classes: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
