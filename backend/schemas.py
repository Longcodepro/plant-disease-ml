"""
Pydantic Schemas for Prediction API
Tham khảo: Designing Machine Learning Systems (Chip Huyen) - Chương 7 & 9 (Prediction Service & Monitoring)
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    class_name: str = Field(..., description="Tên lớp bệnh hoặc tình trạng cây")
    confidence: float = Field(..., description="Độ tin cậy xác suất (0.0 - 1.0)")

class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Nhãn có xác suất cao nhất")
    confidence: float = Field(..., description="Độ tin cậy của nhãn dự đoán cao nhất")
    top_3: List[PredictionItem] = Field(default_factory=list, description="Top 3 dự đoán có xác suất cao nhất")
    inference_time_ms: float = Field(..., description="Thời gian thực thi suy luận (ms)")
    model_version: str = Field(..., description="Phiên bản hoặc tên mô hình đang phục vụ")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
