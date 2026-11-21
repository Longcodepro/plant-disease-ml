# Ultra-Slim ONNX Serving Image (~280 MB thay vì ~4 GB khi cài PyTorch)
# Build từ thư mục gốc dự án:
#   docker build -f backend/Dockerfile -t plant-disease-api:onnx .
# Chạy:
#   docker run -p 8000:8000 plant-disease-api:onnx

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="models/convnext_tiny_khoi_int8.onnx" \
    MAPPING_PATH="models/class_indices.json"

WORKDIR /app

COPY backend/requirements-onnx.txt ./requirements-onnx.txt
RUN pip install --no-cache-dir -r requirements-onnx.txt

COPY backend/__init__.py backend/main.py backend/schemas.py backend/service.py ./backend/
COPY models/class_indices.json models/convnext_tiny_khoi_int8.onnx ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
