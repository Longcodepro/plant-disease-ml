# Ultra-Slim ONNX Serving Image (~280 MB, không PyTorch)
# Tương thích sẵn Hugging Face Spaces (cổng 7860, non-root UID 1000) & Local/Render ($PORT)
#
# Build local:
#   docker build -f backend/Dockerfile -t plantguard-api:onnx .
# Chạy local:
#   docker run -p 8000:8000 -e PORT=8000 plantguard-api:onnx

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MODEL_PATH="models/convnext_tiny_khoi_int8.onnx" \
    MAPPING_PATH="models/class_indices.json" \
    CORS_ORIGINS="*"

# Tạo non-root user UID 1000 (chuẩn bảo mật của Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

COPY backend/requirements-onnx.txt ./requirements-onnx.txt
RUN pip install --no-cache-dir -r requirements-onnx.txt

COPY --chown=user:user backend/__init__.py backend/agronomy.py backend/main.py backend/schemas.py backend/service.py ./backend/
COPY --chown=user:user models/class_indices.json models/convnext_tiny_khoi_int8.onnx ./models/
COPY --chown=user:user data/test ./data/test

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"7860\")}/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
