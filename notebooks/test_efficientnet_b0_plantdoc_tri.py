"""Unit tests for PlantDoc Domain Adaptation module (TDD)."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from efficientnet_b0_plantdoc_tri import (
    ModelEMA,
    apply_mixup_cutmix,
    build_balanced_mixed_weights,
    build_resized_train_cache,
    compute_plantdoc_metrics,
    get_field_train_transforms,
)


def test_field_train_transforms_picklable_and_valid_shape():
    tf = get_field_train_transforms(224)
    restored = pickle.loads(pickle.dumps(tf))
    img = Image.fromarray(np.full((256, 256, 3), 120, dtype=np.uint8), mode="RGB")
    out = restored(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert torch.isfinite(out).all()


def test_apply_mixup_cutmix_produces_valid_soft_targets():
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 5, 12, 37])
    x_mix, y_a, y_b, lam = apply_mixup_cutmix(x, y, cutmix_p=0.5, mixup_p=0.5)
    assert x_mix.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_model_ema_tracks_parameters_smoothly():
    m = torch.nn.Linear(4, 2)
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.fill_(0.0)
    ema = ModelEMA(m, decay=0.9)
    with torch.no_grad():
        m.weight.fill_(2.0)
    ema.update(m)
    # Expected: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    assert torch.allclose(ema.ema.weight, torch.full_like(m.weight, 1.1), atol=1e-5)


def test_build_balanced_mixed_weights_respects_domain_ratio():
    pv_targets = [0, 0, 0, 1]  # 4 lab samples across 2 classes
    pd_targets = [0, 1]        # 2 field samples across 2 classes
    weights = build_balanced_mixed_weights(pv_targets, pd_targets, plantdoc_ratio=0.75)
    assert len(weights) == 6
    pv_mass = float(weights[:4].sum())
    pd_mass = float(weights[4:].sum())
    assert pv_mass == pytest.approx(0.25, rel=1e-4)
    assert pd_mass == pytest.approx(0.75, rel=1e-4)


def test_build_balanced_mixed_weights_caps_singleton_class_oversampling():
    pv_targets = [0] * 10 + [1] * 10
    pd_targets = [0] * 30 + [1]
    weights = build_balanced_mixed_weights(
        pv_targets, pd_targets, plantdoc_ratio=0.80, min_pd_count=15
    )
    assert float(weights[:20].sum()) == pytest.approx(0.20, rel=1e-4)
    assert float(weights[20:].sum()) == pytest.approx(0.80, rel=1e-4)
    assert float(weights[-1]) < float(weights[20:50].sum()) * 0.15


def test_plantdoc_script_uses_valid_for_selection_not_test():
    src = (NOTEBOOKS_DIR / "efficientnet_b0_plantdoc_tri.py").read_text(encoding="utf-8")
    assert 'pd_root / "valid"' in src
    assert "pd_valid_loader" in src
    assert "EFFICIENTNET_B0_TRI_PLANTDOC_VALID" in src
    loop_start = src.index("for local_ep in range(1, n_epochs + 1):")
    loop_end = src.index("total_sec = round(time.perf_counter() - t_all, 2)")
    loop_body = src[loop_start:loop_end]
    assert "pd_valid_loader" in loop_body
    assert "pd_test_loader" not in loop_body


def test_build_resized_train_cache_shrinks_large_images_preserves_classes(tmp_path):
    src = tmp_path / "src"
    cls_dir = src / "Apple___healthy"
    cls_dir.mkdir(parents=True)
    Image.new("RGB", (1200, 600), color=(40, 180, 60)).save(cls_dir / "big.png")
    Image.new("RGB", (200, 150), color=(30, 150, 50)).save(cls_dir / "small.jpg")

    cache = tmp_path / "cache"
    out_dir = build_resized_train_cache(src, cache, max_side=384)
    assert out_dir == cache
    files = sorted((cache / "Apple___healthy").glob("*.jpg"))
    assert len(files) == 2
    sizes = {p.stem: Image.open(p).size for p in files}
    assert sizes["big"] == (384, 192)
    assert sizes["small"] == (200, 150)


def test_compute_plantdoc_metrics_present_classes_only():
    idx_to_class = {0: "A", 1: "B", 2: "C_unseen"}
    y_true = np.array([0, 0, 1, 1])
    # Sample 3 predicts class 2 (absent from PlantDoc ground truth)
    logits = np.array([
        [5.0, 1.0, 0.0],
        [4.0, 2.0, 0.0],
        [0.0, 5.0, 1.0],
        [0.0, 2.0, 5.0],
    ])
    m = compute_plantdoc_metrics(y_true, logits, idx_to_class)
    assert m["num_classes_evaluated"] == 2
    assert m["top1_accuracy"] == pytest.approx(0.75)
    assert m["top3_accuracy"] == pytest.approx(1.0)
    # Class A: TP=2, FP=0, FN=0 -> F1=1.0; Class B: TP=1, FP=0, FN=1 -> F1=2/3=0.6667
    assert m["plantdoc_27class_macro_f1"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, abs=1e-3)
