"""
Automated API Test Suite cho Backend FastAPI (`backend/main.py` & `backend/service.py`).

Chạy:
    pytest -v tests/test_api.py
"""

from __future__ import annotations

import base64
import io
import json
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
os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "convnext_tiny_field_v2_int8.onnx"))
os.environ.setdefault("MAPPING_PATH", str(ROOT / "models" / "class_indices.json"))

from backend.agronomy import format_label, get_advisory  # noqa: E402
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
    assert "CAM" in body["method"] or "Occlusion" in body["method"]
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


def test_agronomy_covers_all_38_classes():
    mapping_path = ROOT / "models" / "class_indices.json"
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    labels = list(data["class_to_idx"].keys())
    assert len(labels) == 38
    for raw_label in labels:
        plant, condition, is_healthy = format_label(raw_label)
        assert plant and condition
        adv = get_advisory(raw_label)
        assert adv["severity"] in ("none", "medium", "high")
        if is_healthy:
            assert adv["severity"] == "none"
        else:
            assert adv["severity"] in ("medium", "high")
        assert adv["summary"]
        assert len(adv["treatment"]) >= 1
        assert len(adv["prevention"]) >= 1


def test_api_health_and_model_info():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert health.json()["model_loaded"] is True

    info = client.get("/api/model-info")
    assert info.status_code == 200
    body = info.json()
    assert body["model_name"] == "ConvNeXt-Tiny"
    assert body["backend_type"] == "onnx"
    assert "INT8" in body["architecture"]
    assert body["num_classes"] == 38
    assert 0.99 <= body["val_macro_f1"] <= 1.0
    assert 0.99 <= body["val_accuracy"] <= 1.0
    assert body["test_accuracy"] == 1.0
    assert body["model_size_mb"] > 0.0
    assert len(body["providers"]) >= 1
    assert body["confidence_threshold"] > 0.0


def test_api_samples_list_fetch_and_path_traversal():
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 34
    assert len(body["samples"]) == body["count"]
    field_samples = [s for s in body["samples"] if s["filename"].startswith("PlantDoc_")]
    assert len(field_samples) >= 1
    first = body["samples"][0]
    assert first["filename"]
    assert first["plant"]
    assert first["condition"]
    assert isinstance(first["is_healthy"], bool)
    assert first["url"] == f"/api/samples/{first['filename']}"

    img_resp = client.get(first["url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/")

    # Path traversal checks
    for bad_name in ("..%2Fbackend%2Fmain.py", "..%5Cmodels%5Cclass_indices.json", "nonexistent_leaf_999.JPG"):
        bad_resp = client.get(f"/api/samples/{bad_name}")
        assert bad_resp.status_code in (400, 404)

    bad_ext = client.get("/api/samples/readme.txt")
    assert bad_ext.status_code == 400


def test_api_predict_uploaded_file_and_xai_layers():
    img_bytes = _make_dummy_png((45, 150, 55))
    r1 = client.post("/api/predict", files={"file": ("leaf_xai.png", img_bytes, "image/png")})
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["plant"]
    assert b1["condition"]
    assert isinstance(b1["is_healthy"], bool)
    assert len(b1["top_3"]) == 3
    assert all(item["plant"] and item["condition"] for item in b1["top_3"])
    assert b1["xai_method"] is not None
    assert b1["advisory"]["severity"] in ("none", "medium", "high")
    assert len(b1["advisory"]["treatment"]) >= 1

    orig_img = Image.open(io.BytesIO(base64.b64decode(b1["original_image_base64"])))
    heat_only = Image.open(io.BytesIO(base64.b64decode(b1["heatmap_only_base64"])))
    overlay_img = Image.open(io.BytesIO(base64.b64decode(b1["heatmap_overlay_base64"])))
    assert orig_img.size == (224, 224)
    assert heat_only.size == (224, 224) and heat_only.mode == "RGBA"
    assert overlay_img.size == (224, 224)

    # Second call hits SHA-256 cache with full XAI layers preserved
    r2 = client.post("/api/predict", files={"file": ("leaf_xai.png", img_bytes, "image/png")})
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["cache_hit"] is True
    assert b2["heatmap_only_base64"] == b1["heatmap_only_base64"]
    assert b2["inference_time_ms"] <= b1["inference_time_ms"]


def test_api_predict_with_sample_name_and_validation():
    resp = client.post("/api/predict", data={"sample_name": "AppleCedarRust1.JPG", "include_xai": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "Apple___Cedar_apple_rust"
    assert body["plant"] == "Apple"
    assert body["is_healthy"] is False
    assert body["advisory"]["severity"] == "medium"

    # Also test query param & include_xai=false
    resp_no_xai = client.post("/api/predict?sample_name=PotatoHealthy1.JPG&include_xai=false")
    assert resp_no_xai.status_code == 200
    body2 = resp_no_xai.json()
    assert body2["predicted_class"] == "Potato___healthy"
    assert body2["is_healthy"] is True
    assert body2["advisory"]["severity"] == "none"
    assert body2["heatmap_overlay_base64"] is None

    # Also test PlantDoc in-the-wild sample prediction
    resp_field = client.post("/api/predict?sample_name=PlantDoc_TomatoLateBlight1.jpg&include_xai=false")
    assert resp_field.status_code == 200
    assert resp_field.json()["predicted_class"] == "Tomato___Late_blight"

    # Reject missing input and invalid sample path traversal
    assert client.post("/api/predict").status_code == 400
    assert client.post("/api/predict", data={"sample_name": "../backend/main.py"}).status_code == 400


def test_api_predict_batch_and_metrics_history():
    test_dir = ROOT / "data" / "test"
    candidates = sorted(test_dir.glob("*.JPG"))[:2]
    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in candidates]

    resp = client.post("/api/predict-batch", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["avg_time_per_image_ms"] > 0
    for item in body["results"]:
        assert item["plant"]
        assert item["condition"]
        assert isinstance(item["is_healthy"], bool)
        assert item["advisory"]["severity"] in ("none", "medium", "high")

    metrics_resp = client.get("/api/metrics")
    assert metrics_resp.status_code == 200
    m = metrics_resp.json()
    assert isinstance(m["history"], list)
    assert len(m["history"]) >= 1
    entry = m["history"][-1]
    for key in ("timestamp", "latency_ms", "confidence", "predicted_class", "plant", "condition", "cache_hit", "is_uncertain"):
        assert key in entry
