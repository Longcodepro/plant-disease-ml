"""Unit tests for EfficientNet-B0 training module (TDD)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from efficientnet_b0_tri import (
    build_efficientnet_b0,
    get_top_confusions,
    save_competition_checkpoint,
    set_backbone_trainable,
)


def test_model_output_shape_and_classifier():
    model = build_efficientnet_b0(num_classes=38, pretrained=False, dropout=0.3)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 38)


def test_freeze_and_unfreeze_backbone():
    model = build_efficientnet_b0(num_classes=38, pretrained=False)

    set_backbone_trainable(model, trainable=False)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    set_backbone_trainable(model, trainable=True)
    assert all(p.requires_grad for p in model.parameters())


def test_save_and_reload_competition_checkpoint(tmp_path):
    model = build_efficientnet_b0(num_classes=38, pretrained=False)
    metrics = {
        "accuracy": 0.9912,
        "macro_f1": 0.9905,
        "weighted_f1": 0.9911,
        "latency_per_image_ms": 1.25,
        "model_size_mb": 15.5,
    }
    ckpt_path = tmp_path / "efficientnet_b0_tri.pth"
    saved = save_competition_checkpoint(
        model=model,
        metrics=metrics,
        save_path=ckpt_path,
        author="Tri",
        model_name="efficientnet_b0",
    )
    assert saved.exists()

    ckpt = torch.load(saved, map_location="cpu", weights_only=True)
    for key in ("model_name", "author", "num_classes", "val_accuracy", "val_f1_macro", "model_state_dict"):
        assert key in ckpt, f"Missing required checkpoint key: {key}"
    assert ckpt["model_name"] == "efficientnet_b0"
    assert ckpt["author"] == "Tri"
    assert ckpt["num_classes"] == 38
    assert ckpt["val_accuracy"] == pytest.approx(0.9912)
    assert ckpt["val_f1_macro"] == pytest.approx(0.9905)

    reloaded = build_efficientnet_b0(num_classes=ckpt["num_classes"], pretrained=False)
    reloaded.load_state_dict(ckpt["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        out = reloaded(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 38)


def test_get_top_confusions_ranking():
    idx_to_class = {0: "ClassA", 1: "ClassB", 2: "ClassC"}
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 0, 0])
    # Misclassifications: (2->0): 3, (0->1): 2, (1->2): 1
    confusions = get_top_confusions(y_true, y_pred, idx_to_class, top_k=2)
    assert len(confusions) == 2
    assert confusions[0]["true_class"] == "ClassC"
    assert confusions[0]["pred_class"] == "ClassA"
    assert confusions[0]["count"] == 3
    assert confusions[1]["true_class"] == "ClassA"
    assert confusions[1]["pred_class"] == "ClassB"
    assert confusions[1]["count"] == 2


def test_class_indices_match_dataset():
    mapping_path = ROOT_DIR / "models" / "class_indices.json"
    train_dir = ROOT_DIR / "data" / "train"
    assert mapping_path.exists()
    assert train_dir.exists()

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    expected_classes = [mapping["idx_to_class"][str(i)] for i in range(38)]
    actual_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    assert actual_classes == expected_classes


def test_no_unsafe_torch_load_weights_only_false():
    for rel in ("notebooks/efficientnet_b0_tri.py", "notebooks/03_efficientnet_b0_tri.ipynb"):
        text = (ROOT_DIR / rel).read_text(encoding="utf-8")
        assert "weights_only=False" not in text, f"Unsafe weights_only=False found in {rel}"

    for ckpt_rel in ("models/efficientnet_b0_tri.pth", "models/baseline1_simple_cnn.pth"):
        ckpt_path = ROOT_DIR / ckpt_rel
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in ckpt


def test_notebook_has_comparison_and_telemetry_charts_and_consistent_summary():
    nb_path = ROOT_DIR / "notebooks" / "03_efficientnet_b0_tri.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    all_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    all_md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    all_outputs = "\n".join(
        "".join(o.get("text", ""))
        for c in nb["cells"]
        if c["cell_type"] == "code"
        for o in c.get("outputs", [])
    )

    assert "efficientnet_b0_tri_gpu_telemetry.csv" in all_code
    assert "So sánh Chất lượng Dự đoán" in all_code
    assert "Nhật ký Phần cứng RTX 3050" in all_code
    assert "36.39 Wh" in all_outputs or "36,39 Wh" in all_outputs
    # Ensure summary markdown does not hardcode contradictory "~14 ms" when output differs
    assert "~14 ms" not in all_md

