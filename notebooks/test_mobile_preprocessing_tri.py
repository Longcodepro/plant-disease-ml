"""Unit tests for mobile preprocessing & stress-test transforms (TDD)."""

import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from mobile_preprocessing_tri import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    DeterministicGaussianNoisePIL,
    DeterministicLightingShiftPIL,
    DeterministicOcclusionPIL,
    MobileLeafPreprocessor,
    RandomGaussianNoisePIL,
    build_stress_scenarios,
    extract_leaf_base_id,
    get_mobile_eval_transforms,
    get_mobile_robust_train_transforms,
    split_unseen_valid_indices,
)


def _make_synthetic_leaf_pil(width: int = 224, height: int = 224, blurry: bool = False) -> Image.Image:
    """Create a synthetic green leaf image with brownish disease spots and vein texture."""
    rng = np.random.default_rng(123)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Green leaf background with slight gradient
    img[:, :, 0] = 55
    img[:, :, 1] = 145
    img[:, :, 2] = 45

    # Draw crisp veins and leaf spot lesions
    cv2.line(img, (20, height // 2), (width - 20, height // 2), (90, 185, 70), 2)
    for y in range(40, height - 40, 32):
        cv2.line(img, (width // 2, y), (width - 30, y - 18), (85, 175, 65), 1)
    cv2.circle(img, (90, 90), 18, (135, 95, 40), -1)
    cv2.circle(img, (150, 140), 14, (150, 110, 45), -1)
    # Subtle texture
    tex = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + tex, 0, 255).astype(np.uint8)

    if blurry:
        img = cv2.GaussianBlur(img, (9, 9), 3.0)

    return Image.fromarray(img, mode="RGB")


def _laplacian_var_l(pil_img: Image.Image) -> float:
    lab = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2LAB)
    return float(cv2.Laplacian(lab[:, :, 0], cv2.CV_64F).var())


def test_imagenet_constants():
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]


def test_extract_leaf_base_id_uuid_and_kaggle_suffixes():
    uuid = "00075aa8-d81a-4184-8541-b692b78d398a"
    assert extract_leaf_base_id(f"data/train/Apple___Apple_scab/{uuid}___FREC_Scab 3335.JPG") == uuid
    assert extract_leaf_base_id(f"{uuid}___FREC_Scab 3335_90deg.JPG") == uuid
    assert extract_leaf_base_id(f"{uuid}___FREC_Scab 3335_180deg.JPG") == uuid
    assert extract_leaf_base_id(f"{uuid}___FREC_Scab 3335_270deg.JPG") == uuid
    assert extract_leaf_base_id(f"{uuid}___FREC_Scab 3335_new30degFlipLR.JPG") == uuid
    assert extract_leaf_base_id(f"{uuid}___FREC_Scab 3335_flipTB.JPG") == uuid
    assert extract_leaf_base_id("RS_Rust 1564.JPG") == extract_leaf_base_id("RS_Rust 1564_flipLR.JPG")


def test_split_unseen_valid_indices_on_real_dataset():
    train_dir = ROOT_DIR / "data" / "train"
    valid_dir = ROOT_DIR / "data" / "valid"
    if not train_dir.exists() or not valid_dir.exists():
        pytest.skip("Dataset data/train or data/valid not present")

    train_ds = datasets.ImageFolder(str(train_dir))
    valid_ds = datasets.ImageFolder(str(valid_dir))
    unseen_idx, sibling_idx = split_unseen_valid_indices(train_ds.samples, valid_ds.samples)

    assert len(unseen_idx) == 6108
    assert len(sibling_idx) == 11464
    assert len(unseen_idx) + len(sibling_idx) == len(valid_ds.samples)
    assert set(unseen_idx).isdisjoint(set(sibling_idx))

    train_ids = {extract_leaf_base_id(p) for p, _ in train_ds.samples}
    for idx in unseen_idx[:200]:
        assert extract_leaf_base_id(valid_ds.samples[idx][0]) not in train_ids
    for idx in sibling_idx[:200]:
        assert extract_leaf_base_id(valid_ds.samples[idx][0]) in train_ids


def test_preprocessor_output_rgb_size_sharpness_and_color_stability():
    preproc = MobileLeafPreprocessor()
    assert preproc.clahe_clip_limit == pytest.approx(1.5)
    assert preproc.blur_threshold == pytest.approx(120.0)
    assert preproc.sharp_amount_blurry == pytest.approx(0.9)
    assert preproc.sharp_amount_clean == pytest.approx(0.0)
    assert preproc.denoise_strength is True

    blurry = _make_synthetic_leaf_pil(224, 224, blurry=True)
    clean = _make_synthetic_leaf_pil(224, 224, blurry=False)

    out_blurry = preproc(blurry)
    out_clean = preproc(clean)

    assert isinstance(out_blurry, Image.Image)
    assert out_blurry.mode == "RGB"
    assert out_blurry.size == (224, 224)
    assert out_clean.mode == "RGB"
    assert out_clean.size == (224, 224)

    # Sharpness (Laplacian variance) on blurry image must increase clearly
    var_before = _laplacian_var_l(blurry)
    var_after = _laplacian_var_l(out_blurry)
    assert var_after > var_before * 1.3, f"Expected sharper output: before={var_before:.2f}, after={var_after:.2f}"

    # LAB A/B channels must remain stable (disease color preserved)
    lab_before = cv2.cvtColor(np.asarray(clean), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_after = cv2.cvtColor(np.asarray(out_clean), cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_diff_a = float(np.mean(np.abs(lab_after[:, :, 1] - lab_before[:, :, 1])))
    mean_diff_b = float(np.mean(np.abs(lab_after[:, :, 2] - lab_before[:, :, 2])))
    assert mean_diff_a < 2.0, f"Channel A shifted too much: {mean_diff_a:.2f}"
    assert mean_diff_b < 2.0, f"Channel B shifted too much: {mean_diff_b:.2f}"


def test_preprocessor_reduces_sensor_noise_and_sharpens_heavy_blur():
    preproc = MobileLeafPreprocessor()
    clean = _make_synthetic_leaf_pil(224, 224, blurry=False)

    # Sensor noise check: bilateral filter before CLAHE must suppress high-frequency grain
    noisy = DeterministicGaussianNoisePIL(sigma=18.0, seed=42)(clean)
    denoised = preproc(noisy)
    no_denoise = MobileLeafPreprocessor(denoise_strength=False)(noisy)
    lap_noisy = _laplacian_var_l(noisy)
    lap_denoised = _laplacian_var_l(denoised)
    lap_no_denoise = _laplacian_var_l(no_denoise)
    assert lap_denoised < lap_noisy * 0.65, (
        f"Expected bilateral denoise to suppress sensor noise: noisy={lap_noisy:.2f}, denoised={lap_denoised:.2f}"
    )
    assert lap_denoised < lap_no_denoise * 0.5, (
        f"Expected denoise_strength=True to cut noise vs False: denoised={lap_denoised:.2f}, no_denoise={lap_no_denoise:.2f}"
    )

    # Heavy blur check: adaptive unsharp mask must sharpen blurry leaf
    heavy_blur = transforms.GaussianBlur(kernel_size=9, sigma=3.0)(clean)
    sharpened = preproc(heavy_blur)
    lap_blur = _laplacian_var_l(heavy_blur)
    lap_sharp = _laplacian_var_l(sharpened)
    assert lap_sharp > lap_blur * 1.3, (
        f"Expected unsharp mask to sharpen heavy blur: blur={lap_blur:.2f}, sharp={lap_sharp:.2f}"
    )


def test_preprocessor_speed_under_5ms_per_image():
    preproc = MobileLeafPreprocessor()
    imgs = [_make_synthetic_leaf_pil(224, 224, blurry=(i % 2 == 0)) for i in range(50)]

    # Warm-up
    _ = preproc(imgs[0])

    t0 = time.perf_counter()
    for im in imgs:
        _ = preproc(im)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_img = elapsed_ms / len(imgs)
    print(f"[BENCH] MobileLeafPreprocessor avg: {ms_per_img:.3f} ms/img (50 images)")
    assert ms_per_img < 5.0, f"Too slow: {ms_per_img:.3f} ms/img"


def test_all_transforms_and_composes_are_picklable_and_deterministic():
    img = _make_synthetic_leaf_pil(224, 224, blurry=False)

    transforms_to_test = [
        MobileLeafPreprocessor(),
        DeterministicGaussianNoisePIL(sigma=18.0, seed=42),
        RandomGaussianNoisePIL(sigma_min=5.0, sigma_max=18.0, p=0.35),
        DeterministicLightingShiftPIL(brightness_factor=0.55, contrast_factor=0.75),
        DeterministicOcclusionPIL(frac=0.22),
        get_mobile_robust_train_transforms(image_size=224, use_preprocessor=True),
        get_mobile_robust_train_transforms(image_size=224, use_preprocessor=False),
        get_mobile_eval_transforms(image_size=224, use_preprocessor=True),
        get_mobile_eval_transforms(image_size=224, use_preprocessor=False),
    ]

    for tfm in transforms_to_test:
        blob = pickle.dumps(tfm)
        restored = pickle.loads(blob)
        out = restored(img)
        if isinstance(tfm, transforms.Compose):
            assert isinstance(out, torch.Tensor)
            assert out.shape == (3, 224, 224)
        else:
            assert isinstance(out, Image.Image)
            assert out.mode == "RGB"
            assert out.size == (224, 224)

    # Determinism checks
    noise_tfm = DeterministicGaussianNoisePIL(sigma=18.0, seed=42)
    n1 = np.asarray(noise_tfm(img))
    n2 = np.asarray(noise_tfm(img))
    assert np.array_equal(n1, n2)
    assert not np.array_equal(n1, np.asarray(img))

    occ_tfm = DeterministicOcclusionPIL(frac=0.22)
    o1 = np.asarray(occ_tfm(img))
    o2 = np.asarray(occ_tfm(img))
    assert np.array_equal(o1, o2)
    occluded_frac = float(np.mean(np.all(o1 != np.asarray(img), axis=-1)))
    assert 0.15 <= occluded_frac <= 0.28, f"Unexpected occluded area frac: {occluded_frac:.3f}"


def test_build_stress_scenarios_has_exact_7_scenarios_and_pickles():
    expected_keys = [
        "clean",
        "blur_mild",
        "blur_heavy",
        "harsh_lighting",
        "sensor_noise",
        "occlusion_22pct",
        "combined_mobile",
    ]
    img = _make_synthetic_leaf_pil(256, 256, blurry=False)

    for use_preproc in (False, True):
        scenarios = build_stress_scenarios(image_size=224, use_preprocessor=use_preproc)
        assert list(scenarios.keys()) == expected_keys
        for name, compose in scenarios.items():
            restored = pickle.loads(pickle.dumps(compose))
            tensor = restored(img)
            assert isinstance(tensor, torch.Tensor), f"{name} did not return Tensor"
            assert tensor.shape == (3, 224, 224), f"{name} wrong shape: {tensor.shape}"
            assert torch.isfinite(tensor).all(), f"{name} produced non-finite values"
