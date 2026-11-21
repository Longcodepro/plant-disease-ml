#!/usr/bin/env bash
# ==============================================================================
# Đẩy Backend FastAPI + ONNX INT8 + 33 ảnh mẫu lên Hugging Face Spaces (Docker SDK)
#
# Cách dùng:
#   1. Tạo 1 Docker Space trống (Blank) trên https://huggingface.co/new-space
#   2. Lấy Write Token tại https://huggingface.co/settings/tokens
#   3. Chạy:
#        HF_TOKEN="hf_xxx" ./scripts/deploy_hf_space.sh <hf_username>/<space_name>
#
# Ví dụ:
#   HF_TOKEN="hf_abc123" ./scripts/deploy_hf_space.sh khoipham/plantguard-ai-api
# ==============================================================================

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Cách dùng: HF_TOKEN=\"hf_xxx\" ./scripts/deploy_hf_space.sh <hf_username>/<space_name>"
    exit 1
fi

SPACE_REPO="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] Thiếu biến môi trường HF_TOKEN (Write token từ https://huggingface.co/settings/tokens)."
    exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[ERROR] Hugging Face Spaces yêu cầu 'git-lfs' cho file model > 10 MB (.onnx)."
    echo "Cài nhanh trên Mac: brew install git-lfs && git lfs install"
    exit 1
fi

STAGE_DIR="$(mktemp -d)"
cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT

echo "[1/4] Đang clone Space repo https://huggingface.co/spaces/$SPACE_REPO ..."
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_REPO}" "$STAGE_DIR/space"

SPACE_DIR="$STAGE_DIR/space"
cd "$SPACE_DIR"
git lfs install --local

echo "[2/4] Đang chuẩn bị Dockerfile, Backend, ONNX INT8 (32.3 MB) và 33 ảnh mẫu..."
rm -rf backend models data Dockerfile
mkdir -p backend models data/test

cp "$ROOT_DIR/backend/Dockerfile" "$SPACE_DIR/Dockerfile"
cp "$ROOT_DIR/backend/requirements-onnx.txt" "$SPACE_DIR/backend/"
cp "$ROOT_DIR/backend/__init__.py" \
   "$ROOT_DIR/backend/agronomy.py" \
   "$ROOT_DIR/backend/main.py" \
   "$ROOT_DIR/backend/schemas.py" \
   "$ROOT_DIR/backend/service.py" "$SPACE_DIR/backend/"
cp "$ROOT_DIR/models/class_indices.json" \
   "$ROOT_DIR/models/convnext_tiny_khoi_int8.onnx" "$SPACE_DIR/models/"
cp "$ROOT_DIR/data/test/"* "$SPACE_DIR/data/test/"

# Hugging Face Spaces YAML metadata (Docker SDK trên cổng 7860)
cat > "$SPACE_DIR/README.md" <<EOF
---
title: PlantGuard AI API
emoji: 🌿
colorFrom: green
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# PlantGuard AI — FastAPI ONNX INT8 Backend
API chẩn đoán 38 bệnh lá cây PlantVillage bằng ConvNeXt-Tiny ONNX INT8 + Exact ONNX CAM/Grad-CAM.
- Health: \`GET /api/health\`
- Docs: \`GET /docs\`
EOF

# Bật Git LFS cho file model .onnx (>10MB)
cat > "$SPACE_DIR/.gitattributes" <<EOF
*.onnx filter=lfs diff=lfs merge=lfs -text
EOF

echo "[3/4] Commit & push lên Hugging Face Spaces..."
git add -A
if git diff --cached --quiet; then
    echo "[INFO] Không có thay đổi mới so với Space hiện tại."
else
    git commit -m "deploy: PlantGuard AI FastAPI + ConvNeXt-Tiny ONNX INT8 + Exact CAM"
    git push origin main
fi

SPACE_SUBDOMAIN="$(echo "$SPACE_REPO" | tr '/_' '--' | tr '[:upper:]' '[:lower:]')"
echo ""
echo "=================================================================="
echo "[4/4] ĐÃ ĐẨY LÊN HUGGING FACE SPACES THÀNH CÔNG!"
echo " • Theo dõi build logs : https://huggingface.co/spaces/${SPACE_REPO}"
echo " • URL API Production  : https://${SPACE_SUBDOMAIN}.hf.space"
echo " • Kiểm tra Health     : https://${SPACE_SUBDOMAIN}.hf.space/api/health"
echo ""
echo " Bước cuối trên Vercel (Root Directory = frontend):"
echo "   VITE_API_BASE_URL=https://${SPACE_SUBDOMAIN}.hf.space"
echo "=================================================================="
