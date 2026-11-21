"""Mobile Preprocessing & Stress-Test Transform Module for Plant Disease ML."""

import math
import os
import random
import re
import zlib
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_KAGGLE_AUG_SUFFIX_RE = re.compile(
    r"(?:_(?:90deg|180deg|270deg|new30degFlipLR|new30deg|newPixel25|flipLR|flipTB|changeLighting[^_]*))+$",
    re.IGNORECASE,
)


def extract_leaf_base_id(filename: str) -> str:
    """Extract canonical leaf identity from a PlantVillage or Kaggle-augmented filename."""
    basename = os.path.basename(str(filename))
    match = _UUID_RE.search(basename)
    if match:
        return match.group(1).lower()

    stem, _ = os.path.splitext(basename)
    cleaned = _KAGGLE_AUG_SUFFIX_RE.sub("", stem)
    return cleaned.lower()


def split_unseen_valid_indices(
    train_samples: Sequence[Union[Tuple[str, int], str]],
    valid_samples: Sequence[Union[Tuple[str, int], str]],
) -> Tuple[List[int], List[int]]:
    """Split validation indices into (unseen_indices, sibling_indices) with zero leaf leakage."""
    train_ids = {
        extract_leaf_base_id(s[0] if isinstance(s, (tuple, list)) else s)
        for s in train_samples
    }

    unseen_indices: List[int] = []
    sibling_indices: List[int] = []
    for idx, sample in enumerate(valid_samples):
        path = sample[0] if isinstance(sample, (tuple, list)) else sample
        if extract_leaf_base_id(path) in train_ids:
            sibling_indices.append(idx)
        else:
            unseen_indices.append(idx)

    return unseen_indices, sibling_indices


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def _add_gaussian_noise_rgb(
    arr: np.ndarray, sigma: float, rng: np.random.Generator
) -> Image.Image:
    """Add zero-mean Gaussian sensor noise to an RGB uint8 array and return a PIL Image."""
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


class MobileLeafPreprocessor:
    """Edge-preserving LAB-L bilateral denoise + CLAHE + adaptive unsharp masking."""

    def __init__(
        self,
        clahe_clip_limit: float = 1.5,
        clahe_grid: int = 8,
        blur_threshold: float = 120.0,
        sharp_amount_blurry: float = 0.9,
        sharp_amount_clean: float = 0.0,
        denoise_strength: bool = True,
    ) -> None:
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid = int(clahe_grid)
        self.blur_threshold = float(blur_threshold)
        self.sharp_amount_blurry = float(sharp_amount_blurry)
        self.sharp_amount_clean = float(sharp_amount_clean)
        self.denoise_strength = bool(denoise_strength)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(_ensure_rgb(img), dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0]

        if self.denoise_strength:
            l_chan = cv2.bilateralFilter(l_chan, d=5, sigmaColor=25, sigmaSpace=25)

        # Create CLAHE per call so the transform instance stays 100% picklable on Windows.
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )
        l_clahe = clahe.apply(l_chan)

        lap_var = float(cv2.Laplacian(l_clahe, cv2.CV_64F).var())
        if lap_var < self.blur_threshold:
            blurred = cv2.GaussianBlur(l_clahe, (5, 5), 0)
            l_out = cv2.addWeighted(
                l_clahe,
                1.0 + self.sharp_amount_blurry,
                blurred,
                -self.sharp_amount_blurry,
                0,
            )
        elif self.sharp_amount_clean > 0.0:
            blurred = cv2.GaussianBlur(l_clahe, (3, 3), 0)
            l_out = cv2.addWeighted(
                l_clahe,
                1.0 + self.sharp_amount_clean,
                blurred,
                -self.sharp_amount_clean,
                0,
            )
        else:
            l_out = l_clahe

        lab_out = lab.copy()
        lab_out[:, :, 0] = l_out
        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_out, mode="RGB")


class DeterministicGaussianNoisePIL:
    """Reproducible per-image Gaussian sensor noise transform for stress testing."""

    def __init__(self, sigma: float = 18.0, seed: int = 42) -> None:
        self.sigma = float(sigma)
        self.seed = int(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(_ensure_rgb(img), dtype=np.uint8)
        content_hash = zlib.crc32(arr.tobytes())
        derived_seed = (content_hash ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        rng = np.random.default_rng(derived_seed)
        return _add_gaussian_noise_rgb(arr, self.sigma, rng)


class RandomGaussianNoisePIL:
    """Stochastic Gaussian sensor noise augmentation for training."""

    def __init__(
        self, sigma_min: float = 5.0, sigma_max: float = 18.0, p: float = 0.35
    ) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb_img = _ensure_rgb(img)
        if random.random() >= self.p:
            return rgb_img
        arr = np.asarray(rgb_img, dtype=np.uint8)
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        rng = np.random.default_rng()
        return _add_gaussian_noise_rgb(arr, sigma, rng)


class DeterministicLightingShiftPIL:
    """Simulate harsh shadow / low-contrast mobile lighting using PIL ImageEnhance."""

    def __init__(
        self, brightness_factor: float = 0.55, contrast_factor: float = 0.75
    ) -> None:
        self.brightness_factor = float(brightness_factor)
        self.contrast_factor = float(contrast_factor)

    def __call__(self, img: Image.Image) -> Image.Image:
        out = _ensure_rgb(img)
        out = ImageEnhance.Brightness(out).enhance(self.brightness_factor)
        out = ImageEnhance.Contrast(out).enhance(self.contrast_factor)
        return out


class DeterministicOcclusionPIL:
    """Simulate finger/leaf occlusion by placing a dark square near the slightly offset center."""

    def __init__(self, frac: float = 0.22) -> None:
        self.frac = float(frac)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(_ensure_rgb(img), dtype=np.uint8).copy()
        h, w = arr.shape[:2]
        side = max(1, min(h, w, int(round(math.sqrt(self.frac * h * w)))))
        cx = int(round(w * 0.54))
        cy = int(round(h * 0.54))
        x0 = max(0, min(w - side, cx - side // 2))
        y0 = max(0, min(h - side, cy - side // 2))
        arr[y0 : y0 + side, x0 : x0 + side] = 24
        return Image.fromarray(arr, mode="RGB")


def get_mobile_robust_train_transforms(
    image_size: int = 224, use_preprocessor: bool = True
) -> transforms.Compose:
    """Build mobile-robust training augmentation pipeline (preprocessor runs after phone noise)."""
    steps = [
        transforms.RandomResizedCrop(
            image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(20),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=7, sigma=(0.8, 3.0))], p=0.50
        ),
        transforms.ColorJitter(
            brightness=0.45, contrast=0.40, saturation=0.25, hue=0.03
        ),
        RandomGaussianNoisePIL(sigma_min=5.0, sigma_max=22.0, p=0.45),
    ]
    if use_preprocessor:
        steps.append(MobileLeafPreprocessor())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.30, scale=(0.05, 0.25), value=0.0),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def get_mobile_eval_transforms(
    image_size: int = 224, use_preprocessor: bool = True
) -> transforms.Compose:
    """Build deterministic evaluation transform pipeline."""
    steps = [transforms.Resize((image_size, image_size))]
    if use_preprocessor:
        steps.append(MobileLeafPreprocessor())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def build_stress_scenarios(
    image_size: int = 224, use_preprocessor: bool = False
) -> Dict[str, transforms.Compose]:
    """Build the 7 deterministic mobile stress-test scenarios (distortions before preprocessor)."""
    scenario_distortions = {
        "clean": [],
        "blur_mild": [transforms.GaussianBlur(kernel_size=5, sigma=1.5)],
        "blur_heavy": [transforms.GaussianBlur(kernel_size=9, sigma=3.0)],
        "harsh_lighting": [DeterministicLightingShiftPIL(0.50, 0.70)],
        "sensor_noise": [DeterministicGaussianNoisePIL(sigma=18.0)],
        "occlusion_22pct": [DeterministicOcclusionPIL(frac=0.22)],
        "combined_mobile": [
            transforms.GaussianBlur(kernel_size=5, sigma=1.5),
            DeterministicLightingShiftPIL(0.60, 0.75),
            DeterministicGaussianNoisePIL(sigma=12.0),
            DeterministicOcclusionPIL(frac=0.15),
        ],
    }

    scenarios: Dict[str, transforms.Compose] = {}
    for name, distortions in scenario_distortions.items():
        steps = [transforms.Resize((image_size, image_size)), *distortions]
        if use_preprocessor:
            steps.append(MobileLeafPreprocessor())
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        scenarios[name] = transforms.Compose(steps)

    return scenarios
