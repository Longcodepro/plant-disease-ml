#!/usr/bin/env python3
"""
Training Script: ResNet-18 (2-Phase Transfer Learning + Evaluation + ONNX Export)
Tác giả: Long
Tuân thủ chuẩn mực đánh giá của nhóm (giống ConvNeXt của Khôi).
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Thêm root vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from notebooks.data_pipeline import build_dataloaders
from notebooks.eval_engine import evaluate_model, print_evaluation_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện ResNet-18 theo chuẩn nhóm")
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2",
                        help="v1: bộ cũ (data/), v2: bộ mới (data_v2/)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Đường dẫn dữ liệu (tự chọn data hoặc data_v2 nếu không truyền)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-p1", type=int, default=2, help="Warmup epochs cho Head")
    parser.add_argument("--epochs-p2", type=int, default=4, help="Fine-tuning epochs toàn mạng")
    parser.add_argument("--lr-p1", type=float, default=1e-3)
    parser.add_argument("--lr-p2-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-p2-head", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-onnx", action="store_true", default=True)
    parser.add_argument("--quantize", action="store_true", default=True)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = str(ROOT_DIR / ("data_v2" if args.dataset == "v2" else "data"))

    print(f"=======================================================")
    print(f"🌿 HUẤN LUYỆN RESNET-18 (TÁC GIẢ: LONG)")
    print(f"=======================================================")
    print(f"📌 Dữ liệu   : {data_dir} (Bộ {args.dataset.upper()})")
    print(f"⚡ Thiết bị  : {device}")
    print(f"📊 Batch size: {args.batch_size} | Epochs: P1={args.epochs_p1}, P2={args.epochs_p2}")

    train_loader, valid_loader, idx_to_class, class_weights = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=4 if device.type == "cuda" else 2,
        pin_memory=(device.type == "cuda")
    )
    num_classes = len(idx_to_class)

    # Khởi tạo mô hình
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if args.dataset == "v1" else None,
        label_smoothing=args.label_smoothing
    )

    save_pth = ROOT_DIR / "models" / f"resnet18_long_{args.dataset}.pth"
    save_onnx = ROOT_DIR / "models" / f"resnet18_long_{args.dataset}.onnx"
    save_onnx_int8 = ROOT_DIR / "models" / f"resnet18_long_{args.dataset}_int8.onnx"
    save_pth.parent.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = 0.0
    best_state = None

    # PHASE 1: Warmup Head
    print(f"\n=== PHASE 1: WARMUP HEAD ({args.epochs_p1} epochs) ===")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    opt_p1 = optim.AdamW(model.fc.parameters(), lr=args.lr_p1, weight_decay=args.weight_decay)
    for ep in range(1, args.epochs_p1 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt_p1, criterion, scaler, device, use_amp)
        val_m = evaluate_model(model, valid_loader, device=device, criterion=criterion)
        print(f"[P1 {ep}/{args.epochs_p1}] ({time.time()-t0:.1f}s) | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc*100:.2f}% | "
              f"Val Loss {val_m['val_loss']:.4f} Acc {val_m['accuracy']*100:.2f}% Macro-F1 {val_m['macro_f1']*100:.2f}%")
        if val_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            print(f"   ⭐ Lưu Best Checkpoint -> {save_pth.name} (Macro-F1: {best_macro_f1*100:.2f}%)")

    # PHASE 2: Fine-Tuning
    print(f"\n=== PHASE 2: FULL FINE-TUNING ({args.epochs_p2} epochs) ===")
    for p in model.parameters():
        p.requires_grad = True

    backbone_params = [p for name, p in model.named_parameters() if "fc" not in name]
    opt_p2 = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_p2_backbone},
        {"params": model.fc.parameters(), "lr": args.lr_p2_head},
    ], weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=args.epochs_p2, eta_min=1e-6)

    for ep in range(1, args.epochs_p2 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt_p2, criterion, scaler, device, use_amp)
        scheduler.step()
        val_m = evaluate_model(model, valid_loader, device=device, criterion=criterion)
        print(f"[P2 {ep:02d}/{args.epochs_p2:02d}] ({time.time()-t0:.1f}s) | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc*100:.2f}% | "
              f"Val Loss {val_m['val_loss']:.4f} Acc {val_m['accuracy']*100:.2f}% Macro-F1 {val_m['macro_f1']*100:.2f}%")
        if val_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            print(f"   ⭐ Lưu Best Checkpoint -> {save_pth.name} (Macro-F1: {best_macro_f1*100:.2f}%)")

    # Lưu checkpoint chính thức
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(model, valid_loader, device=device, criterion=criterion)
    torch.save({
        "model_name": "resnet18",
        "author": "long",
        "dataset_version": args.dataset,
        "num_classes": num_classes,
        "val_accuracy": final_metrics["accuracy"],
        "val_f1_macro": final_metrics["macro_f1"],
        "metrics": final_metrics,
        "model_state_dict": model.state_dict(),
        "class_indices": idx_to_class
    }, str(save_pth))

    print(f"\n=======================================================")
    print_evaluation_summary(final_metrics, model_name=f"RESNET18_LONG_{args.dataset.upper()}")

    # ONNX Export
    if args.export_onnx:
        print(f"\n🌐 Xuất ONNX sang {save_onnx}...")
        model_cpu = model.to("cpu").eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model_cpu, dummy, str(save_onnx),
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False
        )
        print(f"✅ Đã xuất ONNX: {save_onnx} ({save_onnx.stat().st_size / 1024**2:.1f} MB)")

    # INT8 Quantization
    if args.quantize and args.export_onnx:
        print(f"\n📦 Lượng tử hóa ONNX INT8 sang {save_onnx_int8}...")
        from scripts.quantize_onnx import quantize_model
        quantize_model(save_onnx, save_onnx_int8)
        print(f"✅ Đã xuất INT8: {save_onnx_int8} ({save_onnx_int8.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
