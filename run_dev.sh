#!/usr/bin/env bash
# ==============================================================================
# PlantGuard AI — Khởi chạy song song Backend FastAPI (:8000) & Frontend Vite (:5173)
# Cách dùng:  ./run_dev.sh
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"

# Tự tìm Python virtualenv có uvicorn (ưu tiên ~/.venvs/plant-disease, .venv, hoặc PATH)
PYTHON_BIN=""
for candidate in \
    "${VENV_DIR:-}/bin/python" \
    "$HOME/.venvs/plant-disease/bin/python" \
    "$ROOT_DIR/.venv/bin/python" \
    "$(command -v python3 || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c "import uvicorn, fastapi, onnxruntime" 2>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] Không tìm thấy môi trường Python có đủ uvicorn, fastapi, onnxruntime."
    echo "Hãy cài nhanh 1 lần:"
    echo "  python3 -m venv ~/.venvs/plant-disease && source ~/.venvs/plant-disease/bin/activate"
    echo "  pip install -r backend/requirements-onnx.txt"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "[ERROR] Chưa cài Node.js / npm. Hãy cài Node.js >= 18 rồi chạy lại."
    exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "[INFO] Cài đặt dependencies Frontend (npm install)..."
    (cd "$FRONTEND_DIR" && npm install)
fi

BE_PID=""
FE_PID=""
CLEANED_UP=0

cleanup() {
    [[ "$CLEANED_UP" -eq 1 ]] && return
    CLEANED_UP=1
    echo ""
    echo "[INFO] Đang dừng PlantGuard AI (Backend & Frontend)..."
    [[ -n "$FE_PID" ]] && kill "$FE_PID" 2>/dev/null || true
    [[ -n "$BE_PID" ]] && kill "$BE_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[OK] Đã tắt sạch cả 2 tiến trình."
}
trap cleanup INT TERM EXIT

echo "=================================================================="
echo " PlantGuard AI — Modern Precision Bio-Tech Dev Server"
echo " Backend FastAPI : http://127.0.0.1:8000  (Docs: /docs)"
echo " Frontend React  : http://localhost:5173"
echo " Python Runtime  : $PYTHON_BIN"
echo " Nhấn Ctrl+C để dừng đồng thời cả Backend và Frontend."
echo "=================================================================="

(
    cd "$ROOT_DIR"
    exec "$PYTHON_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
) &
BE_PID=$!

# Đợi Backend sẵn sàng; thoát ngay nếu tiến trình Backend chết sớm
BE_READY=0
for _ in $(seq 1 30); do
    if ! kill -0 "$BE_PID" 2>/dev/null; then
        echo "[ERROR] Backend FastAPI thoát bất thường trước khi sẵn sàng."
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null 2>&1; then
        BE_READY=1
        break
    fi
    sleep 0.5
done

if [[ "$BE_READY" -ne 1 ]]; then
    echo "[ERROR] Backend FastAPI không phản hồi /api/health sau 15 giây."
    exit 1
fi

(
    cd "$FRONTEND_DIR"
    exec npm run dev -- --host localhost --port 5173
) &
FE_PID=$!

wait "$BE_PID" "$FE_PID"
