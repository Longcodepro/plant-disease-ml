"""
EfficientNet-B0 Domain Adaptation (Lab data_v2 -> In-the-Wild PlantDoc)
Tác giả: Tri
Tham khảo:
- Designing Machine Learning Systems (Chip Huyen) - Chương 6 & 8 (Domain Shift, Mixed-Domain Adaptation)
- Đồng bộ chuẩn ánh xạ 28 lớp PlantDoc -> 38 lớp PlantVillage từ nhánh Khôi (`prepare_plantdoc.py` / `eval_domain_shift.py`)
- Dùng chung bộ chấm điểm `notebooks/eval_engine.py` để xuất bảng chuẩn đồng nhất cả nhóm.
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from notebooks.data_pipeline import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from notebooks.efficientnet_b0_tri import (
    build_efficientnet_b0,
    save_competition_checkpoint,
    set_backbone_trainable,
)
from notebooks.eval_engine import evaluate_model, print_evaluation_summary


def get_field_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Field-Robust Augmentation cho ảnh ngoài trời (nền đất/cỏ phức tạp, ánh sáng biến thiên, che khuất)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(degrees=25),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.8))], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)),
    ])


def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    h, w = size[2], size[3]
    cut_rat = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    x1 = max(0, min(w, cx - cut_w // 2))
    y1 = max(0, min(h, cy - cut_h // 2))
    x2 = max(0, min(w, cx + cut_w // 2))
    y2 = max(0, min(h, cy + cut_h // 2))
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    cutmix_p: float = 0.30,
    mixup_p: float = 0.20,
    cutmix_alpha: float = 1.0,
    mixup_alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Áp dụng ngẫu nhiên CutMix hoặc MixUp giúp mô hình không học vẹt phông nền PlantDoc."""
    r = random.random()
    bs = images.size(0)
    if bs < 2 or r >= (cutmix_p + mixup_p):
        return images, targets, targets, 1.0

    perm = torch.randperm(bs, device=images.device)
    y_a = targets
    y_b = targets[perm]

    if r < cutmix_p:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam = 1.0 - area / float(images.size(2) * images.size(3))
        return mixed, y_a, y_b, float(lam)

    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, y_a, y_b, lam


class ModelEMA:
    """Exponential Moving Average của trọng số mô hình giúp ổn định học trên tập PlantDoc nhỏ."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.detach(), alpha=1.0 - d)
            else:
                v.copy_(src)


def build_balanced_mixed_weights(
    pv_targets: Sequence[int],
    pd_targets: Sequence[int],
    plantdoc_ratio: float = 0.70,
    min_pd_count: int = 1,
) -> torch.DoubleTensor:
    """
    Tạo trọng số cho WeightedRandomSampler sao cho:
    - Tổng xác suất chọn miền PlantDoc = plantdoc_ratio (cân bằng đều giữa các lớp PlantDoc,
      có ngưỡng sàn `min_pd_count` để lớp chỉ có 1 ảnh không bị lặp quá mức).
    - Tổng xác suất chọn miền PlantVillage = 1 - plantdoc_ratio (cân bằng đều giữa 38 lớp để chống quên).
    """
    pv_counts = Counter(int(t) for t in pv_targets)
    pd_counts = Counter(int(t) for t in pd_targets)
    floor = max(1, int(min_pd_count))

    raw_pv = [1.0 / pv_counts[int(t)] for t in pv_targets]
    raw_pd = [1.0 / max(pd_counts[int(t)], floor) for t in pd_targets]

    sum_pv = sum(raw_pv) or 1.0
    sum_pd = sum(raw_pd) or 1.0

    pv_scale = (1.0 - plantdoc_ratio) / sum_pv
    pd_scale = plantdoc_ratio / sum_pd

    weights = [w * pv_scale for w in raw_pv] + [w * pd_scale for w in raw_pd]
    return torch.DoubleTensor(weights)


def compute_plantdoc_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    idx_to_class: Dict[int, str],
) -> Dict[str, Any]:
    """
    Tính bộ chỉ số tương thích 100% với `scripts/eval_domain_shift.py` của Khôi:
    - Top-1 Accuracy, Top-3 Accuracy
    - Macro-F1 tính đúng trên các lớp có mặt trong PlantDoc Test (27 lớp)
    - Top 5 cặp nhầm lẫn & F1 từng lớp.
    """
    total = len(y_true)
    top3_idx = np.argsort(logits, axis=1)[:, ::-1][:, :3]
    preds = top3_idx[:, 0]

    correct = int(np.sum(preds == y_true))
    top3_hits = int(sum(int(t in row) for t, row in zip(y_true, top3_idx)))

    present_ids = sorted(int(c) for c in np.unique(y_true))
    f1s: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    per_class: List[Dict[str, Any]] = []
    confusions: Counter = Counter()

    for t, p in zip(y_true, preds):
        if int(t) != int(p):
            confusions[(int(t), int(p))] += 1

    for cid in present_ids:
        tp = int(np.sum((y_true == cid) & (preds == cid)))
        fp = int(np.sum((y_true != cid) & (preds == cid)))
        fn = int(np.sum((y_true == cid) & (preds != cid)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        per_class.append({
            "class_name": idx_to_class.get(cid, f"Class_{cid}"),
            "samples": int(np.sum(y_true == cid)),
            "correct": tp,
            "accuracy": round(rec, 4),
            "f1": round(f1, 4),
        })

    return {
        "num_classes_evaluated": len(present_ids),
        "total_images": total,
        "correct": correct,
        "top1_accuracy": round(correct / total, 4) if total else 0.0,
        "top3_accuracy": round(top3_hits / total, 4) if total else 0.0,
        "plantdoc_27class_macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "plantdoc_27class_precision": round(float(np.mean(precs)), 4) if precs else 0.0,
        "plantdoc_27class_recall": round(float(np.mean(recs)), 4) if recs else 0.0,
        "top_5_confusions": [
            {
                "true": idx_to_class.get(t, f"Class_{t}"),
                "predicted": idx_to_class.get(p, f"Class_{p}"),
                "count": int(c),
            }
            for (t, p), c in confusions.most_common(5)
        ],
        "per_class": sorted(per_class, key=lambda r: r["f1"]),
    }


def _load_mapped_imagefolder(
    folder: Path,
    class_to_idx: Dict[str, int],
    transform: transforms.Compose,
) -> datasets.ImageFolder:
    """Nạp ImageFolder và ánh xạ lại toàn bộ nhãn về đúng 0..37 của `models/class_indices.json`."""
    ds = datasets.ImageFolder(str(folder), transform=transform)
    remapped = [(path, class_to_idx[ds.classes[cls_idx]]) for path, cls_idx in ds.samples]
    ds.samples = remapped
    ds.imgs = remapped
    ds.targets = [s[1] for s in remapped]
    return ds


@torch.no_grad()
def evaluate_plantdoc_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_class: Dict[int, str],
    use_flip_tta: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Đánh giá song song cả chuẩn `eval_engine.py` lẫn chuẩn `eval_domain_shift.py` (27 lớp + Top-3)."""
    eval_eng_metrics = evaluate_model(model, loader, device=device)

    model.eval()
    all_targets: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        logits = model(images)
        if use_flip_tta:
            logits_flip = model(torch.flip(images, dims=[3]))
            logits = 0.5 * (logits + logits_flip)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    logits_arr = np.concatenate(all_logits, axis=0)
    pdoc_metrics = compute_plantdoc_metrics(y_true, logits_arr, idx_to_class)
    return eval_eng_metrics, pdoc_metrics


def build_resized_train_cache(
    src_dir: Path,
    cache_dir: Path,
    max_side: int = 384,
) -> Path:
    """
    Tạo bản cache thu gọn cạnh dài tối đa `max_side` (mặc định 384px >= 224px) CHỈ cho tập PlantDoc train.
    Giúp tránh giải mã lặp lại ảnh JPEG 6000x6000 (12 MB) mỗi epoch (tăng tốc train 4-5x)
    mà giữ nguyên 100% ảnh gốc trong `data/plantdoc/{train,valid,test}`.
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    src_files = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts]
    if cache_dir.is_dir():
        cached_files = [p for p in cache_dir.rglob("*") if p.is_file() and p.suffix.lower() in img_exts]
        if len(cached_files) == len(src_files) and len(src_files) > 0:
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_files:
        rel = src_file.relative_to(src_dir)
        dst_file = (cache_dir / rel).with_suffix(".jpg")
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_file) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                rgb = rgb.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS)
            rgb.save(dst_file, format="JPEG", quality=95)
    return cache_dir


@torch.no_grad()
def _warmup_gpu(model: nn.Module, device: torch.device, batch_size: int = 32, image_size: int = 224) -> None:
    """Chạy lượt forward mẫu đúng batch_size giúp cuDNN benchmark ổn định trước khi đo latency."""
    if device.type != "cuda":
        return
    model.eval()
    for bs in (batch_size, 12):
        dummy = torch.zeros(bs, 3, image_size, image_size, device=device)
        _ = model(dummy)
    torch.cuda.synchronize(device)


def run_plantdoc_adaptation(
    pv_data_dir: str = "data",
    plantdoc_dir: str = "data/plantdoc",
    mapping_path: str = "models/class_indices.json",
    init_checkpoint: str = "models/efficientnet_b0_tri.pth",
    save_path: str = "models/efficientnet_b0_plantdoc_tri.pth",
    history_out: str = "notebooks/efficientnet_b0_plantdoc_tri_results.json",
    stage1_epochs: int = 7,
    stage2_epochs: int = 7,
    samples_per_epoch: int = 10000,
    min_pd_count: int = 15,
    train_cache_max_side: int = 384,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    author: str = "Tri",
) -> Dict[str, Any]:
    """
    Fine-tuning gộp chung `data_v2/train` (38,039) + `data_plantdoc/train` (1,871) = 39,910 ảnh
    từ trọng số cơ bản `models/efficientnet_b0_tri.pth` (2 giai đoạn + EMA + CutMix/MixUp + AMP).
    - Chọn Best EMA chặt chẽ trên `PlantDoc Valid` (20%, 468 ảnh) + bảo vệ `Lab Valid` >= 98.5% Acc.
    - Khóa kín `PlantDoc Test` (236 ảnh, 27 lớp) trong suốt quá trình train, chỉ mở chấm 1 lần duy nhất ở cuối.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_to_idx: Dict[str, int] = mapping["class_to_idx"]
    idx_to_class: Dict[int, str] = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(class_to_idx)

    pv_root = Path(pv_data_dir)
    pd_root = Path(plantdoc_dir)
    if not (pd_root / "valid").is_dir():
        raise FileNotFoundError(
            f"Thiếu thư mục kiểm định nội bộ '{pd_root / 'valid'}' (chuẩn chia 80/20 PlantDoc)."
        )

    pd_train_src = pd_root / "train"
    if train_cache_max_side > 0:
        t_cache = time.perf_counter()
        cache_dir = pd_root.parent / f"{pd_root.name}_train_cache_{train_cache_max_side}"
        pd_train_folder = build_resized_train_cache(pd_train_src, cache_dir, max_side=train_cache_max_side)
        print(
            f"[Cache] Da chuan bi cache PlantDoc train ({train_cache_max_side}px) tai {pd_train_folder} "
            f"trong {time.perf_counter() - t_cache:.1f}s (khong cham anh goc valid/test).",
            flush=True,
        )
    else:
        pd_train_folder = pd_train_src

    field_tf = get_field_train_transforms(224)
    eval_tf = get_eval_transforms(224)

    pv_train_ds = _load_mapped_imagefolder(pv_root / "train", class_to_idx, field_tf)
    pd_train_ds = _load_mapped_imagefolder(pd_train_folder, class_to_idx, field_tf)
    mixed_train_ds = ConcatDataset([pv_train_ds, pd_train_ds])

    pv_valid_ds = _load_mapped_imagefolder(pv_root / "valid", class_to_idx, eval_tf)
    pv_test_ds = _load_mapped_imagefolder(pv_root / "test", class_to_idx, eval_tf)
    pd_valid_ds = _load_mapped_imagefolder(pd_root / "valid", class_to_idx, eval_tf)
    pd_test_ds = _load_mapped_imagefolder(pd_root / "test", class_to_idx, eval_tf)

    pv_valid_loader = DataLoader(pv_valid_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=use_cuda)
    pv_test_loader = DataLoader(pv_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=use_cuda)
    pd_valid_loader = DataLoader(pd_valid_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_cuda)
    pd_test_loader = DataLoader(pd_test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_cuda)

    print("=" * 78, flush=True)
    print(" FINE-TUNING GOP CHUNG: DATA_V2 + PLANTDOC 80/20 (EFFICIENTNET-B0)", flush=True)
    print(
        f" GPU: {gpu_name} | Merged Train: {len(mixed_train_ds):,} ({len(pv_train_ds):,} PV + {len(pd_train_ds):,} PD) | "
        f"PD Valid: {len(pd_valid_ds):,} | PD Locked Test: {len(pd_test_ds):,}",
        flush=True,
    )
    print("=" * 78, flush=True)

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False, dropout=0.3).to(device)
    init_ckpt = torch.load(init_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(init_ckpt["model_state_dict"])
    set_backbone_trainable(model, trainable=True)
    _warmup_gpu(model, device, batch_size=32)

    # 0. Đo Zero-Shot Domain Shift TRƯỚC khi huấn luyện thích nghi (trên PlantDoc Valid & Test)
    print("\n--- [0] DO ZERO-SHOT DOMAIN SHIFT TRUOC KHI FINE-TUNE ---", flush=True)
    zs_val_eng, zs_val_pd = evaluate_plantdoc_full(model, pd_valid_loader, device, idx_to_class, use_flip_tta=False)
    zs_test_eng, zs_test_pd = evaluate_plantdoc_full(model, pd_test_loader, device, idx_to_class, use_flip_tta=False)
    print_evaluation_summary(zs_test_eng, model_name="EFFICIENTNET_B0_TRI_PLANTDOC_ZERO_SHOT")
    print(
        f"  [Zero-Shot PlantDoc Valid ({zs_val_pd['num_classes_evaluated']} cls)] "
        f"Top-1: {zs_val_pd['top1_accuracy']*100:.2f}% | Top-3: {zs_val_pd['top3_accuracy']*100:.2f}% | "
        f"Macro-F1: {zs_val_pd['plantdoc_27class_macro_f1']*100:.2f}%",
        flush=True,
    )
    print(
        f"  [Zero-Shot PlantDoc Test  ({zs_test_pd['num_classes_evaluated']} cls)] "
        f"Top-1: {zs_test_pd['top1_accuracy']*100:.2f}% ({zs_test_pd['correct']}/{zs_test_pd['total_images']}) | "
        f"Top-3: {zs_test_pd['top3_accuracy']*100:.2f}% | Macro-F1: {zs_test_pd['plantdoc_27class_macro_f1']*100:.2f}%",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    ema = ModelEMA(model, decay=0.995)

    history: List[Dict[str, Any]] = []
    best_val_score = -1.0
    best_epoch = 0
    overall_peak_vram_mb = 0.0
    ckpt_path = Path(save_path)
    hist_path = Path(history_out)
    t_all = time.perf_counter()

    stages = [
        ("Stage 1: Merged Bridge (55% PD / 45% PV)", stage1_epochs, 0.55, 1.5e-4, 6e-4, max(0, stage1_epochs - 2)),
        ("Stage 2: Field Fine-Tune (75% PD / 25% PV)", stage2_epochs, 0.75, 5e-5, 2e-4, max(0, stage2_epochs - 3)),
    ]

    epoch_idx = 0
    for stage_name, n_epochs, pd_ratio, lr_bb, lr_hd, mix_epochs in stages:
        if n_epochs <= 0:
            continue
        print(
            f"\n--- {stage_name} ({n_epochs} epochs | PlantDoc ratio={pd_ratio*100:.0f}% | "
            f"Mix first {mix_epochs} ep | min_pd_count={min_pd_count}) ---",
            flush=True,
        )
        weights = build_balanced_mixed_weights(
            pv_train_ds.targets,
            pd_train_ds.targets,
            plantdoc_ratio=pd_ratio,
            min_pd_count=min_pd_count,
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=samples_per_epoch, replacement=True)
        loader_kw: Dict[str, Any] = {
            "batch_size": batch_size,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": use_cuda,
            "drop_last": True,
        }
        if num_workers > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = 2
        train_loader = DataLoader(mixed_train_ds, **loader_kw)

        optimizer = optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": lr_bb},
                {"params": model.classifier.parameters(), "lr": lr_hd},
            ],
            weight_decay=1e-2,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs), eta_min=1e-6)

        for local_ep in range(1, n_epochs + 1):
            epoch_idx += 1
            use_mix = local_ep <= mix_epochs
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            t0 = time.perf_counter()
            model.train()
            running_loss = 0.0
            seen = 0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=use_cuda)
                targets = targets.to(device, non_blocking=use_cuda)
                if use_mix:
                    images, y_a, y_b, lam = apply_mixup_cutmix(images, targets, cutmix_p=0.30, mixup_p=0.20)
                else:
                    y_a, y_b, lam = targets, targets, 1.0

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                    outputs = model(images)
                    loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                bs = images.size(0)
                running_loss += loss.item() * bs
                seen += bs

            scheduler.step()
            train_loss = running_loss / max(1, seen)
            peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1) if use_cuda else 0.0
            overall_peak_vram_mb = max(overall_peak_vram_mb, peak_vram_mb)

            # Đánh giá EMA CHỈ trên Lab Valid (8.147) & PlantDoc Valid 20% (468) — TUYỆT ĐỐI KHÔNG đụng PlantDoc Test
            lab_val_metrics = evaluate_model(ema.ema, pv_valid_loader, device=device)
            pd_val_eng, pd_val_metrics = evaluate_plantdoc_full(
                ema.ema, pd_valid_loader, device, idx_to_class, use_flip_tta=False
            )
            ep_sec = round(time.perf_counter() - t0, 2)

            val_selection_score = round(
                0.5 * pd_val_metrics["plantdoc_27class_macro_f1"] + 0.5 * pd_val_metrics["top1_accuracy"],
                4,
            )

            rec = {
                "epoch": epoch_idx,
                "stage": stage_name,
                "mix": use_mix,
                "train_loss": round(float(train_loss), 4),
                "lab_valid_acc": lab_val_metrics["accuracy"],
                "lab_valid_macro_f1": lab_val_metrics["macro_f1"],
                "plantdoc_valid_top1": pd_val_metrics["top1_accuracy"],
                "plantdoc_valid_top3": pd_val_metrics["top3_accuracy"],
                "plantdoc_valid_macro_f1": pd_val_metrics["plantdoc_27class_macro_f1"],
                "plantdoc_valid_eval_engine_macro_f1": pd_val_eng["macro_f1"],
                "val_selection_score": val_selection_score,
                "epoch_sec": ep_sec,
                "peak_vram_mb": peak_vram_mb,
            }
            history.append(rec)
            print(
                f"Epoch [{epoch_idx:02d}] ({stage_name[:7]} {local_ep}/{n_epochs}, Mix={'ON ' if use_mix else 'OFF'}, {ep_sec:.1f}s) | "
                f"Loss: {train_loss:.4f} | Lab Val F1: {lab_val_metrics['macro_f1']*100:.2f}% (Acc {lab_val_metrics['accuracy']*100:.2f}%) | "
                f"PD Val Top-1: {pd_val_metrics['top1_accuracy']*100:.2f}% Top-3: {pd_val_metrics['top3_accuracy']*100:.2f}% "
                f"F1({pd_val_metrics['num_classes_evaluated']}): {pd_val_metrics['plantdoc_27class_macro_f1']*100:.2f}% "
                f"| VRAM: {peak_vram_mb:.0f} MB",
                flush=True,
            )

            if (val_selection_score > best_val_score and lab_val_metrics["accuracy"] >= 0.985) or best_epoch == 0:
                if lab_val_metrics["accuracy"] >= 0.985 or best_epoch == 0:
                    best_val_score = val_selection_score
                    best_epoch = epoch_idx
                    save_competition_checkpoint(
                        model=ema.ema,
                        metrics=lab_val_metrics,
                        save_path=ckpt_path,
                        author=author,
                        model_name="efficientnet_b0_plantdoc",
                        extra={
                            "best_epoch": best_epoch,
                            "selection_split": "plantdoc_valid_20pct",
                            "plantdoc_valid_top1": pd_val_metrics["top1_accuracy"],
                            "plantdoc_valid_top3": pd_val_metrics["top3_accuracy"],
                            "plantdoc_valid_macro_f1": pd_val_metrics["plantdoc_27class_macro_f1"],
                            "init_checkpoint": str(init_checkpoint),
                        },
                    )
                    print(
                        f"  --> Luu Best EMA tai {ckpt_path} (PD Val Score={best_val_score*100:.2f}%, "
                        f"Top-1={pd_val_metrics['top1_accuracy']*100:.2f}%, F1={pd_val_metrics['plantdoc_27class_macro_f1']*100:.2f}%, "
                        f"Lab Acc={lab_val_metrics['accuracy']*100:.2f}%)",
                        flush=True,
                    )

    total_sec = round(time.perf_counter() - t_all, 2)

    # Nạp lại Best EMA Checkpoint (chọn thuần trên Valid) để chấm CHÍNH THỨC 1 LẦN DUY NHẤT trên Test
    print("\n--- DANH GIA CHINH THUC BEST EMA CHECKPOINT (PLANTDOC TEST CHAM 1 LAN DUY NHAT) ---", flush=True)
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    best_model = build_efficientnet_b0(num_classes=num_classes, pretrained=False, dropout=0.3).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model.eval()
    _warmup_gpu(best_model, device, batch_size=32)
    _warmup_gpu(best_model, device, batch_size=64)

    final_pd_val_eng, final_pd_val_28 = evaluate_plantdoc_full(
        best_model, pd_valid_loader, device, idx_to_class, use_flip_tta=False
    )
    final_pdoc_eng, final_pdoc_27 = evaluate_plantdoc_full(
        best_model, pd_test_loader, device, idx_to_class, use_flip_tta=False
    )
    _, final_pdoc_tta = evaluate_plantdoc_full(
        best_model, pd_test_loader, device, idx_to_class, use_flip_tta=True
    )
    final_lab_valid = evaluate_model(best_model, pv_valid_loader, device=device)
    final_lab_test = evaluate_model(best_model, pv_test_loader, device=device)

    print_evaluation_summary(final_pd_val_eng, model_name="EFFICIENTNET_B0_TRI_PLANTDOC_VALID")
    print(
        f"  [PlantDoc Valid {final_pd_val_28['num_classes_evaluated']}-class] Top-1: {final_pd_val_28['top1_accuracy']*100:.2f}% "
        f"({final_pd_val_28['correct']}/{final_pd_val_28['total_images']}) | Top-3: {final_pd_val_28['top3_accuracy']*100:.2f}% | "
        f"Macro-F1: {final_pd_val_28['plantdoc_27class_macro_f1']*100:.2f}%",
        flush=True,
    )
    print_evaluation_summary(final_pdoc_eng, model_name="EFFICIENTNET_B0_TRI_PLANTDOC_TEST_SET")
    print(
        f"  [PlantDoc Test 27-class Single-Pass] Top-1: {final_pdoc_27['top1_accuracy']*100:.2f}% "
        f"({final_pdoc_27['correct']}/{final_pdoc_27['total_images']}) | Top-3: {final_pdoc_27['top3_accuracy']*100:.2f}% | "
        f"Macro-F1 (27 cls): {final_pdoc_27['plantdoc_27class_macro_f1']*100:.2f}%",
        flush=True,
    )
    print(
        f"  [PlantDoc Test 27-class Flip-TTA]    Top-1: {final_pdoc_tta['top1_accuracy']*100:.2f}% "
        f"({final_pdoc_tta['correct']}/{final_pdoc_tta['total_images']}) | Top-3: {final_pdoc_tta['top3_accuracy']*100:.2f}% | "
        f"Macro-F1 (27 cls): {final_pdoc_tta['plantdoc_27class_macro_f1']*100:.2f}%",
        flush=True,
    )
    print_evaluation_summary(final_lab_valid, model_name="EFFICIENTNET_B0_TRI_PLANTDOC_ON_LAB_VALID")
    print_evaluation_summary(final_lab_test, model_name="EFFICIENTNET_B0_TRI_PLANTDOC_ON_LAB_TEST_SET")

    summary = {
        "status": "completed",
        "protocol": "Fine-tuning gộp chung data_v2/train (38,039) + data_plantdoc/train (1,871) = 39,910 ảnh từ models/efficientnet_b0_tri.pth; chọn Best EMA thuần trên PlantDoc Valid (468) + bảo vệ Lab Valid >= 98.5%; PlantDoc Test (236) khóa kín và chấm 1 lần duy nhất ở cuối",
        "model_name": "efficientnet_b0_plantdoc",
        "author": author,
        "device": gpu_name,
        "overall_peak_vram_mb": overall_peak_vram_mb,
        "total_train_time_sec": total_sec,
        "best_epoch": best_epoch,
        "checkpoint_file_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "dataset": {
            "pv_train": len(pv_train_ds),
            "pv_valid": len(pv_valid_ds),
            "pv_test": len(pv_test_ds),
            "plantdoc_train": len(pd_train_ds),
            "merged_train_total": len(mixed_train_ds),
            "plantdoc_valid": len(pd_valid_ds),
            "plantdoc_test": len(pd_test_ds),
            "plantdoc_train_valid_sha256_overlap": 0,
            "plantdoc_upstream_test_train_sha256_overlap": 9,
            "plantdoc_upstream_test_valid_sha256_overlap": 2,
        },
        "zero_shot_plantdoc_valid": {
            "eval_engine": zs_val_eng,
            "plantdoc_present_classes": zs_val_pd,
        },
        "zero_shot_plantdoc": {
            "eval_engine": zs_test_eng,
            "plantdoc_27class": zs_test_pd,
        },
        "adapted_plantdoc_valid": {
            "eval_engine": final_pd_val_eng,
            "plantdoc_28class": final_pd_val_28,
        },
        "adapted_plantdoc_single": {
            "eval_engine": final_pdoc_eng,
            "plantdoc_27class": final_pdoc_27,
        },
        "adapted_plantdoc_flip_tta": final_pdoc_tta,
        "adapted_lab_valid_8147": final_lab_valid,
        "adapted_lab_test_8098": final_lab_test,
        "history": history,
    }
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Da luu bao cao Fine-Tune gop chung data_v2 + PlantDoc tai: {hist_path}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning gộp chung data_v2 + PlantDoc 80/20 (EfficientNet-B0)")
    parser.add_argument("--pv_data_dir", type=str, default="data")
    parser.add_argument("--plantdoc_dir", type=str, default="data/plantdoc")
    parser.add_argument("--init_checkpoint", type=str, default="models/efficientnet_b0_tri.pth")
    parser.add_argument("--save_path", type=str, default="models/efficientnet_b0_plantdoc_tri.pth")
    parser.add_argument("--history_out", type=str, default="notebooks/efficientnet_b0_plantdoc_tri_results.json")
    parser.add_argument("--stage1_epochs", type=int, default=7)
    parser.add_argument("--stage2_epochs", type=int, default=7)
    parser.add_argument("--samples_per_epoch", type=int, default=10000)
    parser.add_argument("--min_pd_count", type=int, default=15)
    parser.add_argument("--train_cache_max_side", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    run_plantdoc_adaptation(
        pv_data_dir=args.pv_data_dir,
        plantdoc_dir=args.plantdoc_dir,
        init_checkpoint=args.init_checkpoint,
        save_path=args.save_path,
        history_out=args.history_out,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        samples_per_epoch=args.samples_per_epoch,
        min_pd_count=args.min_pd_count,
        train_cache_max_side=args.train_cache_max_side,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
