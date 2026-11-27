#!/usr/bin/env python3
"""
Domain Adaptation & Robust Augmentation Fine-Tuning Pipeline for ResNet-18
Tác giả: Long

Kỹ thuật:
1. Outdoor Physical Augmentations: ColorJitter (nắng/mây), GaussianBlur (mờ nhòe), RandomAffine (lá quăn góc chụp), RandomErasing (rách lá).
2. Domain Expansion: Kết hợp dữ liệu phòng lab với tập lá thực địa đã bóc tách bằng YOLO (data_plantdoc_leaves/train).
3. Fine-tuning từ trọng số ResNet-18 đã hội tụ (v1, v2, v3) với Cosine Annealing Learning Rate.
4. Tự động xuất bản ONNX FP32 và nén INT8 quantization.
"""

from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDocLeafDataset(Dataset):
    """Nạp các phiến lá con đã bóc tách từ PlantDoc train, ánh xạ đúng về 38 lớp PlantVillage."""
    def __init__(self, root_dir: Path, class_to_idx: dict[str, int], transform=None):
        self.samples = []
        self.transform = transform
        
        for cls_dir in root_dir.iterdir():
            if not cls_dir.is_dir() or cls_dir.name not in class_to_idx:
                continue
            idx = class_to_idx[cls_dir.name]
            for img_file in cls_dir.glob("*.*"):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_file, idx))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


class SimpleImageDataset(Dataset):
    """Nạp dataset dạng ImageFolder chuẩn (train/valid)."""
    def __init__(self, root_dir: Path, class_to_idx: dict[str, int], transform=None):
        self.samples = []
        self.transform = transform
        for cls_dir in root_dir.iterdir():
            if not cls_dir.is_dir() or cls_dir.name not in class_to_idx:
                continue
            idx = class_to_idx[cls_dir.name]
            for img_file in cls_dir.glob("*.*"):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_file, idx))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target


def build_robust_transforms():
    """Bộ chuyển đổi giả lập điều kiện quang học và vật lý ngoài tự nhiên."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        # Giả lập nắng gắt, mây râm và đổi sắc màu
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08),
        # Giả lập máy ảnh rung tay, mờ nét
        transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Giả lập sâu ăn, rách lá hoặc lá bị che khuất
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random")
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    return train_transform, val_transform


def train_domain_adaptation(
    version: str,
    base_checkpoint: Path,
    dataset_version: str,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = "cuda"
):
    print(f"=======================================================")
    print(f"🌿 HUẤN LUYỆN TĂNG CƯỜNG THỰC ĐỊA (DOMAIN ADAPTATION)")
    print(f"=======================================================")
    print(f"📌 Mô hình gốc       : {base_checkpoint.name}")
    print(f"📦 Bộ dữ liệu nền    : {dataset_version}")
    print(f"⚡ Thiết bị          : {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"📊 Epochs: {epochs}  | Batch Size: {batch_size} | Base LR: {lr}")

    mapping_path = ROOT_DIR / "models" / "class_indices.json"
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    num_classes = len(class_to_idx)

    train_tf, val_tf = build_robust_transforms()

    # 1. Base lab dataset
    base_data_dir = ROOT_DIR / f"data_{dataset_version}" / "train"
    if not base_data_dir.exists():
        base_data_dir = ROOT_DIR / "data_v3" / "train"
    base_dataset = SimpleImageDataset(base_data_dir, class_to_idx, transform=train_tf)
    print(f"📁 Dữ liệu phòng lab : {len(base_dataset)} ảnh ({base_data_dir})")

    # 2. Field leaf crops (tách bằng YOLO)
    field_leaves_dir = ROOT_DIR / "data_plantdoc_leaves" / "train"
    datasets_to_combine = [base_dataset]
    if field_leaves_dir.exists():
        field_dataset = PlantDocLeafDataset(field_leaves_dir, class_to_idx, transform=train_tf)
        print(f"🍃 Dữ liệu lá thực địa (YOLO): {len(field_dataset)} lá con ({field_leaves_dir})")
        # Nhân đôi trọng số thực địa để mô hình thích ứng nhanh
        datasets_to_combine.extend([field_dataset, field_dataset])

    combined_train_ds = ConcatDataset(datasets_to_combine)
    print(f"🔥 Tổng mẫu huấn luyện kết hợp: {len(combined_train_ds)} ảnh")

    # 3. Validation set (PlantDoc test set)
    val_dir = ROOT_DIR / "data_plantdoc" / "test"
    val_dataset = SimpleImageDataset(val_dir, class_to_idx, transform=val_tf)
    print(f"🎯 Tập kiểm thử thực địa (PlantDoc Test): {len(val_dataset)} ảnh")

    train_loader = DataLoader(
        combined_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4 if device == "cuda" else 2, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4 if device == "cuda" else 2, pin_memory=(device == "cuda")
    )

    # 4. Khởi tạo mạng từ checkpoint gốc
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    print("✅ Đã nạp thành công trọng số từ checkpoint gốc.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    tag = f"resnet18_long_{dataset_version}_adapted_{version}"
    save_pth = ROOT_DIR / "models" / f"{tag}.pth"
    save_onnx = ROOT_DIR / "models" / f"{tag}.onnx"
    save_onnx_int8 = ROOT_DIR / "models" / f"{tag}_int8.onnx"

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        running_loss, correct_tr, total_tr = 0.0, 0, 0
        
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct_tr += (preds == targets).sum().item()
            total_tr += targets.size(0)
            
        scheduler.step()
        train_loss = running_loss / total_tr
        train_acc = (correct_tr / total_tr) * 100.0

        # Validate on PlantDoc Test
        model.eval()
        val_loss, correct_val, top3_val, total_val = 0.0, 0, 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == targets).sum().item()
                
                # Top 3
                _, top3_preds = outputs.topk(3, dim=1)
                top3_val += sum(t.item() in top3_preds[i] for i, t in enumerate(targets))
                total_val += targets.size(0)
                
        val_acc = (correct_val / total_val) * 100.0
        val_top3 = (top3_val / total_val) * 100.0
        val_loss = val_loss / total_val
        ep_duration = time.time() - t_ep

        print(f"[{epoch}/{epochs}] ({ep_duration:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | PlantDoc Test Loss: {val_loss:.4f} Top-1: {val_acc:.2f}% Top-3: {val_top3:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "dataset_version": dataset_version,
                "version": version,
                "plantdoc_val_acc": val_acc,
                "plantdoc_top3": val_top3
            }, save_pth)
            print(f"   ⭐ Saved Best Model Checkpoint -> {save_pth.name} (Top-1: {val_acc:.2f}%)")

    # Export ONNX & INT8
    print(f"\n⚡ Xuất ONNX sang: {save_onnx.name}")
    model.load_state_dict(best_weights)
    model.to("cpu").eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, str(save_onnx),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        dynamo=False
    )

    # Quantize to INT8
    print(f"📦 Đang nén INT8: {save_onnx_int8.name}...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(save_onnx), str(save_onnx_int8),
        weight_type=QuantType.QUInt8
    )
    print(f"✅ Hoàn tất! File đã tạo: {save_onnx_int8.name} ({save_onnx_int8.stat().st_size / 1024**2:.1f} MB)")
    return save_pth, save_onnx_int8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="run_01")
    parser.add_argument("--dataset", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--base-ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.base_ckpt is None:
        args.base_ckpt = ROOT_DIR / "models" / f"resnet18_long_{args.dataset}_run_01.pth"
    else:
        args.base_ckpt = Path(args.base_ckpt)

    train_domain_adaptation(
        version=args.version,
        base_checkpoint=args.base_ckpt,
        dataset_version=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=dev
    )
