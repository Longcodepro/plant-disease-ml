"""
Automated API Test Suite cho Backend FastAPI (`backend/main.py` & `backend/service.py`).

Chạy:
    pytest -v tests/test_api.py
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Mặc định test trực tiếp bằng model ONNX INT8 (không cần cài PyTorch)
os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "convnext_tiny_khoi_int8.onnx"))
os.environ.setdefault("MAPPING_PATH", str(ROOT / "models" / "class_indices.json"))

from backend.main import MAX_UPLOAD_BYTES, app, service  # noqa: E402

client = TestClient(app)


def _make_dummy_png(color=(34, 139, 34)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_lazy_import_no_torch_for_onnx():
    """Chứng minh chạy ONNX không cần nạp PyTorch vào bộ nhớ."""
    assert service.backend_type == "onnx"
    assert "torch" not in sys.modules


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert "ONNX" in body["model_version"]


def test_predict_real_test_image():
    sample = ROOT / "data" / "test" / "AppleCedarRust1.JPG"
    img_bytes = sample.read_bytes() if sample.exists() else _make_dummy_png()
    mime = "image/jpeg" if sample.exists() else "image/png"

    resp = client.post("/predict", files={"file": ("leaf.jpg", img_bytes, mime)})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["top_3"]) == 3
    assert 0.0 < body["confidence"] <= 1.0
    assert body["inference_time_ms"] > 0
    assert isinstance(body["is_uncertain"], bool)
    if sample.exists():
        assert body["predicted_class"] == "Apple___Cedar_apple_rust"
        assert body["is_uncertain"] is False
        assert body["warning"] is None


def test_predict_batch_multiple_images():
    test_dir = ROOT / "data" / "test"
    candidates = sorted(test_dir.glob("*.JPG"))[:3] if test_dir.exists() else []
    if len(candidates) == 3:
        files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in candidates]
    else:
        files = [("files", (f"leaf_{i}.png", _make_dummy_png(), "image/png")) for i in range(3)]

    resp = client.post("/predict/batch", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert body["total_inference_time_ms"] > 0
    assert len(body["results"]) == 3
    for item in body["results"]:
        assert item["predicted_class"]
        assert len(item["top_3"]) == 3


def test_predict_rejects_non_image_file():
    resp = client.post("/predict", files={"file": ("note.txt", b"hello world", "text/plain")})
    assert resp.status_code == 400


def test_predict_rejects_corrupt_image_bytes():
    resp = client.post("/predict", files={"file": ("broken.jpg", b"not-a-valid-jpeg", "image/jpeg")})
    assert resp.status_code == 400


def test_predict_rejects_oversized_file():
    oversized = b"\xff" * (MAX_UPLOAD_BYTES + 1)
    resp = client.post("/predict", files={"file": ("huge.jpg", oversized, "image/jpeg")})
    assert resp.status_code == 413


def test_uncertain_warning_when_below_threshold():
    old_thr = service.confidence_threshold
    try:
        # Đặt ngưỡng 0.9999 để kiểm chứng cờ is_uncertain + warning
        service.confidence_threshold = 0.9999
        resp = client.post("/predict", files={"file": ("noise.png", _make_dummy_png((128, 128, 128)), "image/png")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_uncertain"] is True
        assert body["warning"] is not None
    finally:
        service.confidence_threshold = old_thr


def test_sha256_cache_hit_returns_submillisecond():
    img = _make_dummy_png((50, 160, 60))
    r1 = client.post("/predict", files={"file": ("leaf_c.png", img, "image/png")})
    r2 = client.post("/predict", files={"file": ("leaf_c.png", img, "image/png")})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["cache_hit"] is False
    assert r2.json()["cache_hit"] is True
    assert r2.json()["predicted_class"] == r1.json()["predicted_class"]
    assert r2.json()["inference_time_ms"] < r1.json()["inference_time_ms"]


def test_explain_heatmap_endpoint():
    sample = ROOT / "data" / "test" / "AppleCedarRust1.JPG"
    img_bytes = sample.read_bytes() if sample.exists() else _make_dummy_png()
    mime = "image/jpeg" if sample.exists() else "image/png"

    resp = client.post("/explain", files={"file": ("leaf.jpg", img_bytes, mime)})
    assert resp.status_code == 200
    body = resp.json()
    assert "Occlusion Sensitivity" in body["method"]
    assert body["explain_time_ms"] > 0
    if sample.exists():
        assert body["predicted_class"] == "Apple___Cedar_apple_rust"
    raw_png = base64.b64decode(body["heatmap_base64"])
    heatmap = Image.open(io.BytesIO(raw_png))
    assert heatmap.size == (224, 224)


def test_metrics_telemetry_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] >= 1
    assert body["total_images_predicted"] >= 1
    assert body["cache_hits"] >= 1
    assert 0.0 <= body["cache_hit_rate"] <= 1.0
    assert 0.0 <= body["uncertain_rate"] <= 1.0
    assert body["latency_ms"]["p95"] >= body["latency_ms"]["p50"] >= 0.0
    assert len(body["top_predicted_classes"]) >= 1
