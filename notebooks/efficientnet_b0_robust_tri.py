"""
EfficientNet-B0 Robust v2 (Mobile Preprocessing + Degradation-Aware Fine-Tuning)
Tác giả: Tri
Mô tả:
- Khởi tạo từ trọng số v1 (`models/efficientnet_b0_tri.pth`, nạp an toàn `weights_only=True`).
- Huấn luyện tinh chỉnh trên ảnh lá được mô phỏng nhiễu điện thoại (mờ rung tay, tối/chói sáng,
  nhiễu hạt cảm biến, che khuất) đi qua bộ tiền xử lý `MobileLeafPreprocessor` (LAB-CLAHE + Adaptive Unsharp Mask)
  để chống Training-Serving Skew (Chip Huyen Ch. 5 & 8).
- KHÔNG chỉnh sửa `notebooks/data_pipeline.py` hay `notebooks/eval_engine.py` dùng chung của nhóm.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from notebooks.data_pipeline import compute_class_weights
from notebooks.efficientnet_b0_tri import (
    build_efficientnet_b0,
    save_competition_checkpoint,
    set_backbone_trainable,
    train_one_epoch_amp,
)
from notebooks.eval_engine import evaluate_model, print_evaluation_summary
from notebooks.mobile_preprocessing_tri import (
    get_mobile_eval_transforms,
    get_mobile_robust_train_transforms,
)


def run_robust_finetuning(
    data_dir: str = "data",
    init_checkpoint: str = "models/efficientnet_b0_tri.pth",
    save_path: str = "models/efficientnet_b0_robust_tri.pth",
    history_out: str = "notebooks/efficientnet_b0_robust_tri_results.json",
    epochs: int = 3,
    batch_size: int = 32,
    num_workers: int = 4,
    lr_backbone: float = 5e-5,
    lr_head: float = 2e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    seed: int = 42,
    author: str = "Tri",
) -> Dict[str, Any]:
    """Tinh chỉnh EfficientNet-B0 với MobileLeafPreprocessor và tăng cường nhiễu điện thoại."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    train_path = Path(data_dir) / "train"
    valid_path = Path(data_dir) / "valid"

    train_ds = datasets.ImageFolder(
        root=str(train_path),
        transform=get_mobile_robust_train_transforms(image_size=224, use_preprocessor=True),
    )
    valid_ds = datasets.ImageFolder(
        root=str(valid_path),
        transform=get_mobile_eval_transforms(image_size=224, use_preprocessor=True),
    )

    class_weights = compute_class_weights(train_ds)
    num_classes = len(train_ds.classes)

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_cuda, drop_last=False)

    print("=" * 68, flush=True)
    print(" TINH CHINH EFFICIENTNET-B0 ROBUST V2 (MOBILE PREPROC + BLUR/NOISE)", flush=True)
    print(f" Thiet bi: {gpu_name} | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print("=" * 68, flush=True)

    model = build_efficientnet_b0(num_classes=num_classes, pretrained=False, dropout=0.3).to(device)
    init_path = Path(init_checkpoint)
    if init_path.exists():
        init_ckpt = torch.load(init_path, map_location=device, weights_only=True)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"[OK] Da nap trong so khoi tao tu v1: {init_path}", flush=True)
    else:
        raise FileNotFoundError(f"Khong tim thay checkpoint khoi tao: {init_path}")

    set_backbone_trainable(model, trainable=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: List[Dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    overall_peak_vram_mb = 0.0
    ckpt_out = Path(save_path)
    hist_path = Path(history_out)
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        ep_lr = float(optimizer.param_groups[0]["lr"])
        ep_t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
        train_sec = time.perf_counter() - ep_t0
        val_metrics = evaluate_model(model, valid_loader, device=device, criterion=criterion)
        scheduler.step()
        ep_total_sec = time.perf_counter() - ep_t0

        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
            if use_cuda
            else 0.0
        )
        overall_peak_vram_mb = max(overall_peak_vram_mb, peak_vram_mb)
        throughput = len(train_ds) / train_sec if train_sec > 0 else 0.0

        rec = {
            "epoch": ep,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": val_metrics.get("val_loss", 0.0),
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "train_sec": round(train_sec, 2),
            "epoch_total_sec": round(ep_total_sec, 2),
            "train_throughput_img_s": round(throughput, 1),
            "peak_vram_mb": peak_vram_mb,
            "lr": round(ep_lr, 7),
        }
        history.append(rec)
        print(
            f"Robust Epoch [{ep:02d}/{epochs:02d}] ({ep_total_sec:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% ({throughput:.0f} img/s) | "
            f"Val Acc: {val_metrics['accuracy']*100:.2f}% Macro-F1: {val_metrics['macro_f1']*100:.2f}% | "
            f"Peak VRAM: {peak_vram_mb:.0f} MB",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = ep
            save_competition_checkpoint(
                model=model,
                metrics=val_metrics,
                save_path=ckpt_out,
                author=author,
                model_name="efficientnet_b0_robust",
                extra={
                    "best_epoch": best_epoch,
                    "preprocessing": "MobileLeafPreprocessor(LAB-CLAHE + Adaptive Unsharp Mask)",
                    "init_checkpoint": str(init_path),
                },
            )
            print(f"  --> Luu Best Robust Checkpoint tai {ckpt_out} (Macro-F1: {best_macro_f1*100:.2f}%)", flush=True)

    total_sec = round(time.perf_counter() - t_start, 2)
    best_ckpt = torch.load(ckpt_out, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = evaluate_model(model, valid_loader, device=device, criterion=criterion)
    final_metrics["checkpoint_file_size_mb"] = round(ckpt_out.stat().st_size / (1024 * 1024), 2)
    print_evaluation_summary(final_metrics, model_name=f"EfficientNet-B0 Robust v2 ({author})")

    summary = {
        "status": "completed",
        "model_name": "efficientnet_b0_robust",
        "author": author,
        "preprocessing": "MobileLeafPreprocessor(LAB-CLAHE + Adaptive Unsharp Mask)",
        "device": gpu_name,
        "overall_peak_vram_mb": overall_peak_vram_mb,
        "total_train_time_sec": total_sec,
        "best_epoch": best_epoch,
        "final_metrics_with_preproc": final_metrics,
        "history": history,
    }
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Da luu ket qua Robust v2 tai: {hist_path}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tinh chỉnh EfficientNet-B0 Robust v2 cho ảnh điện thoại")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--init_checkpoint", type=str, default="models/efficientnet_b0_tri.pth")
    parser.add_argument("--save_path", type=str, default="models/efficientnet_b0_robust_tri.pth")
    parser.add_argument("--history_out", type=str, default="notebooks/efficientnet_b0_robust_tri_results.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_robust_finetuning(
        data_dir=args.data_dir,
        init_checkpoint=args.init_checkpoint,
        save_path=args.save_path,
        history_out=args.history_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
