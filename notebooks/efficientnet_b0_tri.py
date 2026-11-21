"""
EfficientNet-B0 (Transfer Learning 2-Phase + AMP fp16) — Plant Disease Classification
Tác giả: Tri
Tham khảo:
- Designing Machine Learning Systems (Chip Huyen) - Chương 4, 5, 6, 7 (Fair Offline Eval, Class Imbalance, Latency & Size Trade-off)
- Hands-On Machine Learning (Aurélien Géron) - Chương 13, 14 (Transfer Learning 2-Phase, Data Augmentation)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from notebooks.data_pipeline import build_dataloaders
from notebooks.eval_engine import evaluate_model, print_evaluation_summary


def build_efficientnet_b0(
    num_classes: int = 38,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Khởi tạo EfficientNet-B0 với lớp phân loại tùy chỉnh cho 38 tình trạng cây trồng."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Đóng băng hoặc mở khóa phần thân trích xuất đặc trưng (features), luôn giữ classifier học."""
    for param in model.features.parameters():
        param.requires_grad = trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def train_one_epoch_amp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Huấn luyện 1 epoch với Mixed Precision (AMP fp16 trên CUDA) để tiết kiệm VRAM 4 GB."""
    model.train()
    use_amp = device.type == "cuda"
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += batch_size

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thu thập toàn bộ nhãn thật và nhãn dự đoán trên tập Validation (dùng cho phân tích lỗi)."""
    model.eval()
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds)


def get_top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Trả về Top-K cặp (nhãn thật -> nhãn dự đoán sai) xảy ra nhiều nhất."""
    counts: Counter = Counter()
    for t, p in zip(y_true, y_pred):
        ti, pi = int(t), int(p)
        if ti != pi:
            counts[(ti, pi)] += 1

    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], idx_to_class.get(kv[0][0], str(kv[0][0])), idx_to_class.get(kv[0][1], str(kv[0][1]))),
    )[:top_k]

    return [
        {
            "true_class": idx_to_class.get(t_idx, f"Class_{t_idx}"),
            "pred_class": idx_to_class.get(p_idx, f"Class_{p_idx}"),
            "count": int(cnt),
        }
        for (t_idx, p_idx), cnt in ranked
    ]


def save_competition_checkpoint(
    model: nn.Module,
    metrics: Dict[str, Any],
    save_path: Union[str, Path],
    author: str = "Tri",
    model_name: str = "efficientnet_b0",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Lưu checkpoint theo đúng quy chuẩn thi đấu nhóm trong AI_COLLABORATION_GUIDE.md."""
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = int(model.classifier[-1].out_features)

    payload: Dict[str, Any] = {
        "model_name": model_name,
        "author": author,
        "num_classes": num_classes,
        "val_accuracy": float(metrics["accuracy"]),
        "val_f1_macro": float(metrics["macro_f1"]),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, out_path)
    return out_path


def evaluate_baseline1(
    valid_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    baseline_path: Union[str, Path] = "models/baseline1_simple_cnn.pth",
) -> Optional[Dict[str, Any]]:
    """Đánh giá lại Baseline 1 (SimplePlantCNN) trên cùng valid_loader và cùng GPU để so sánh công bằng."""
    ckpt_path = Path(baseline_path)
    if not ckpt_path.exists():
        return None

    from notebooks.baseline1_simple_cnn import SimplePlantCNN

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    num_classes = int(ckpt.get("num_classes", 38))
    baseline_model = SimplePlantCNN(num_classes=num_classes)
    baseline_model.load_state_dict(ckpt["model_state_dict"])
    baseline_model.to(device)
    baseline_model.eval()

    metrics = evaluate_model(baseline_model, valid_loader, device=device, criterion=criterion)
    metrics["checkpoint_file_size_mb"] = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
    return metrics


def _make_even_subset_loader(
    loader: DataLoader,
    fraction: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Lấy mẫu đều trên toàn bộ dataset (phủ đủ 38 lớp) khi chạy dry-run với subset_fraction < 1.0."""
    n_total = len(loader.dataset)
    n_sub = max(batch_size, int(n_total * fraction))
    n_sub = min(n_sub, n_total)
    indices = np.linspace(0, n_total - 1, num=n_sub, dtype=int).tolist()
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_training(
    data_dir: str = "data",
    epochs_phase1: int = 3,
    epochs_phase2: int = 12,
    batch_size: int = 32,
    num_workers: int = 2,
    lr_head_p1: float = 1e-3,
    lr_backbone_p2: float = 1e-4,
    lr_head_p2: float = 5e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    dropout: float = 0.3,
    patience: int = 4,
    subset_fraction: float = 1.0,
    seed: int = 42,
    save_path: str = "models/efficientnet_b0_tri.pth",
    history_out: str = "notebooks/efficientnet_b0_tri_results.json",
    baseline_path: str = "models/baseline1_simple_cnn.pth",
    author: str = "Tri",
) -> Dict[str, Any]:
    """Chạy toàn bộ quy trình huấn luyện 2 pha, đánh giá công bằng và phân tích lỗi."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    total_vram_mb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        if use_cuda
        else 0.0
    )

    print("=" * 64, flush=True)
    print(" KHOI DONG HUAN LUYEN: EFFICIENTNET-B0 (2-PHASE + AMP FP16)", flush=True)
    print(f" Thiet bi: {gpu_name} | VRAM: {total_vram_mb} MB | Batch: {batch_size}", flush=True)
    print("=" * 64, flush=True)

    train_loader, valid_loader, idx_to_class, class_weights = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=224,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    if subset_fraction < 1.0:
        print(f"[*] Dry-run mode: su dung {subset_fraction * 100:.1f}% du lieu (lay deu cac lop)...", flush=True)
        train_loader = _make_even_subset_loader(
            train_loader, subset_fraction, batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda
        )
        valid_loader = _make_even_subset_loader(
            valid_loader, subset_fraction, batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda
        )
    elif num_workers > 0:
        # Giu nguyen dataset/transforms tu data_pipeline.py, bat persistent_workers=True de tranh overhead spawn tren Windows
        train_loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        valid_loader = DataLoader(
            dataset=valid_loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=2,
        )

    num_classes = len(idx_to_class)
    model = build_efficientnet_b0(num_classes=num_classes, pretrained=True, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: List[Dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    overall_peak_vram_mb = 0.0
    total_start = time.perf_counter()
    history_path = Path(history_out)
    ckpt_path = Path(save_path)

    def _run_phase(
        phase_num: int,
        phase_name: str,
        num_epochs: int,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        early_stop_patience: Optional[int],
        start_epoch_idx: int,
    ) -> int:
        nonlocal best_macro_f1, best_epoch, best_metrics, overall_peak_vram_mb
        no_improve = 0
        current_epoch = start_epoch_idx

        print(f"\n--- PHA {phase_num}: {phase_name} ({num_epochs} epochs) ---", flush=True)
        for local_ep in range(1, num_epochs + 1):
            current_epoch += 1
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            ep_start = time.perf_counter()
            train_loss, train_acc = train_one_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
            train_sec = time.perf_counter() - ep_start
            train_throughput = len(train_loader.dataset) / train_sec if train_sec > 0 else 0.0

            val_metrics = evaluate_model(model, valid_loader, device=device, criterion=criterion)
            scheduler.step()

            ep_total_sec = time.perf_counter() - ep_start
            peak_vram_mb = (
                round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
                if use_cuda
                else 0.0
            )
            overall_peak_vram_mb = max(overall_peak_vram_mb, peak_vram_mb)

            rec = {
                "epoch": current_epoch,
                "phase": phase_num,
                "phase_name": phase_name,
                "train_loss": round(float(train_loss), 4),
                "train_acc": round(float(train_acc), 4),
                "val_loss": val_metrics.get("val_loss", 0.0),
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_latency_ms": val_metrics["latency_per_image_ms"],
                "train_sec": round(train_sec, 2),
                "epoch_total_sec": round(ep_total_sec, 2),
                "train_throughput_img_s": round(train_throughput, 1),
                "peak_vram_mb": peak_vram_mb,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 7),
            }
            history.append(rec)

            print(
                f"Epoch [{current_epoch:02d}] (P{phase_num} {local_ep}/{num_epochs}, {ep_total_sec:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% ({train_throughput:.0f} img/s) | "
                f"Val Loss: {rec['val_loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% "
                f"Macro-F1: {val_metrics['macro_f1']*100:.2f}% | Peak VRAM: {peak_vram_mb:.0f} MB",
                flush=True,
            )

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = current_epoch
                best_metrics = dict(val_metrics)
                no_improve = 0
                save_competition_checkpoint(
                    model=model,
                    metrics=val_metrics,
                    save_path=ckpt_path,
                    author=author,
                    model_name="efficientnet_b0",
                    extra={"best_epoch": best_epoch, "dropout": dropout},
                )
                print(
                    f"  --> Luu Best Checkpoint tai {ckpt_path} (Macro-F1: {best_macro_f1*100:.2f}%)",
                    flush=True,
                )
            else:
                no_improve += 1

            # Ghi nhat ky tam sau moi epoch de chong mat du lieu va ho tro theo doi truc tiep
            _write_json(
                history_path,
                {
                    "status": "running",
                    "model_name": "efficientnet_b0",
                    "author": author,
                    "device": gpu_name,
                    "total_vram_mb": total_vram_mb,
                    "overall_peak_vram_mb": overall_peak_vram_mb,
                    "best_epoch": best_epoch,
                    "best_macro_f1": best_macro_f1,
                    "history": history,
                },
            )

            if early_stop_patience is not None and no_improve >= early_stop_patience:
                print(
                    f"  [*] Early Stopping kich hoat sau {early_stop_patience} epoch khong cai thien Macro-F1.",
                    flush=True,
                )
                break

        return current_epoch

    # PHA 1: Warm-up (Khoa backbone, chi train classifier)
    epoch_cursor = 0
    if epochs_phase1 > 0:
        set_backbone_trainable(model, trainable=False)
        opt_p1 = optim.AdamW(model.classifier.parameters(), lr=lr_head_p1, weight_decay=weight_decay)
        sched_p1 = optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=max(1, epochs_phase1), eta_min=1e-4)
        epoch_cursor = _run_phase(1, "Warm-up (Frozen Backbone)", epochs_phase1, opt_p1, sched_p1, None, epoch_cursor)

    # PHA 2: Fine-tuning toan mang voi Differential Learning Rate
    if epochs_phase2 > 0:
        set_backbone_trainable(model, trainable=True)
        opt_p2 = optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": lr_backbone_p2},
                {"params": model.classifier.parameters(), "lr": lr_head_p2},
            ],
            weight_decay=weight_decay,
        )
        sched_p2 = optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=max(1, epochs_phase2), eta_min=1e-6)
        epoch_cursor = _run_phase(
            2, "Fine-tune (Full Unfrozen)", epochs_phase2, opt_p2, sched_p2, patience, epoch_cursor
        )

    total_train_time_sec = round(time.perf_counter() - total_start, 2)

    # Nap lai dung Best Checkpoint de cham diem cuoi cung & phan tich loi
    print("\n--- DANH GIA CUOI CUNG TREN BEST CHECKPOINT ---", flush=True)
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = evaluate_model(model, valid_loader, device=device, criterion=criterion)
    final_metrics["checkpoint_file_size_mb"] = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
    print_evaluation_summary(final_metrics, model_name=f"EfficientNet-B0 ({author})")

    # Phan tich loi & F1 theo tung lop
    y_true, y_pred = collect_predictions(model, valid_loader, device)
    top_confusions = get_top_confusions(y_true, y_pred, idx_to_class, top_k=5)
    present_labels = sorted(set(int(x) for x in np.unique(y_true)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=[idx_to_class[i] for i in present_labels],
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = [
        {
            "class_name": idx_to_class[i],
            "f1_score": round(float(report_dict[idx_to_class[i]]["f1-score"]), 4),
            "precision": round(float(report_dict[idx_to_class[i]]["precision"]), 4),
            "recall": round(float(report_dict[idx_to_class[i]]["recall"]), 4),
            "support": int(report_dict[idx_to_class[i]]["support"]),
        }
        for i in present_labels
    ]
    per_class_f1_sorted = sorted(per_class_f1, key=lambda x: x["f1_score"])

    # So sanh cong bang voi Baseline 1 tren cung valid_loader & GPU
    baseline_metrics = evaluate_baseline1(valid_loader, device=device, criterion=criterion, baseline_path=baseline_path)
    if baseline_metrics is not None:
        print_evaluation_summary(baseline_metrics, model_name="Baseline 1 (SimplePlantCNN)")

    # Cap nhat lai best_metrics chinh xac trong file checkpoint
    save_competition_checkpoint(
        model=model,
        metrics=final_metrics,
        save_path=ckpt_path,
        author=author,
        model_name="efficientnet_b0",
        extra={"best_epoch": best_epoch, "dropout": dropout},
    )

    summary: Dict[str, Any] = {
        "status": "completed",
        "model_name": "efficientnet_b0",
        "author": author,
        "device": gpu_name,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "total_vram_mb": total_vram_mb,
        "overall_peak_vram_mb": overall_peak_vram_mb,
        "config": {
            "data_dir": data_dir,
            "epochs_phase1": epochs_phase1,
            "epochs_phase2": epochs_phase2,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "lr_head_p1": lr_head_p1,
            "lr_backbone_p2": lr_backbone_p2,
            "lr_head_p2": lr_head_p2,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "dropout": dropout,
            "patience": patience,
            "subset_fraction": subset_fraction,
            "seed": seed,
            "amp_fp16": use_cuda,
        },
        "dataset": {
            "num_classes": num_classes,
            "train_samples": len(train_loader.dataset),
            "valid_samples": len(valid_loader.dataset),
        },
        "total_train_time_sec": total_train_time_sec,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
        "baseline1_metrics": baseline_metrics,
        "top_5_confusions": top_confusions,
        "worst_5_classes": per_class_f1_sorted[:5],
        "best_5_classes": list(reversed(per_class_f1_sorted[-5:])),
        "history": history,
    }
    _write_json(history_path, summary)
    print(f"[OK] Da luu bao cao chi tiet tai: {history_path}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện EfficientNet-B0 (2-Phase + AMP fp16)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs_phase1", type=int, default=3)
    parser.add_argument("--epochs_phase2", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--save_path", type=str, default="models/efficientnet_b0_tri.pth")
    parser.add_argument("--history_out", type=str, default="notebooks/efficientnet_b0_tri_results.json")
    parser.add_argument("--baseline_path", type=str, default="models/baseline1_simple_cnn.pth")
    parser.add_argument("--author", type=str, default="Tri")
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patience=args.patience,
        subset_fraction=args.subset_fraction,
        save_path=args.save_path,
        history_out=args.history_out,
        baseline_path=args.baseline_path,
        author=args.author,
    )
