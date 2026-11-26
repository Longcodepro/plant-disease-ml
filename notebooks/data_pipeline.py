"""
Data Pipeline Module: Preprocessing, Data Augmentation & DataLoader Factory
Dự án: Chẩn đoán bệnh trên lá cây nông nghiệp
Tham khảo:
- Hands-On Machine Learning (Aurélien Géron) - Chương 13, 14 (Loading & Preprocessing Data, Deep Computer Vision)
- Designing Machine Learning Systems (Chip Huyen) - Chương 5 (Feature Engineering & Data Leakage Prevention)
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Chuẩn hóa giá trị Pixel theo phân phối chuẩn của ImageNet
# Lý do: Khi sử dụng Transfer Learning (MobileNetV2, ResNet50), backbone đã được pretrain
# trên các kênh RGB với mean và std này. Giữ nguyên chuẩn hóa này giúp weights hội tụ ổn định.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Data Augmentation Pipeline dành RIÊNG cho tập Train.
    Mục tiêu (Hands-On ML Ch. 14):
    - Tăng độ đa dạng của dữ liệu (Regularization chống Overfitting).
    - Mô phỏng điều kiện thực tế của nông dân ngoài đồng ruộng:
      + Lá cây chụp ở nhiều góc xoay khác nhau (RandomRotation).
      + Hướng chụp lật ngang/dọc (RandomHorizontalFlip, RandomVerticalFlip).
      + Điều kiện ánh sáng thay đổi nắng/mây/bóng râm (ColorJitter: brightness, contrast).
      + Góc chụp xa/gần (RandomResizedCrop).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_dataset1_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Pipeline Tiền Xử Lý & Data Augmentation TỐI ƯU RIÊNG CHO DATASET 1 (Chống Data Leakage & Học Vẹt):
    - RandomResizedCrop(scale 0.6 - 1.0): Zoom cận cảnh vào vết bệnh, cắt bỏ viền lá để mô hình không "nhớ mặt" cả chiếc lá.
    - RandomRotation (30 deg) & RandomPerspective: Biến dạng 3D góc chụp.
    - ColorJitter mạnh: Khử phụ thuộc vào màu nền chụp trong nhà kính.
    - RandomErasing (Cutout, p=0.3): Che ngẫu nhiên một mảng nhỏ trên lá, ép mạng nơ-ron phải tìm
      nhiều dấu hiệu bệnh khác nhau thay vì dựa vào 1 vết sẹo/vết rách đặc trưng của chiếc lá đó.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.6, 1.0),
            ratio=(0.85, 1.15)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value="random"
        )
    ])

def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic Preprocessing Pipeline dành cho tập Validation & Test (và Production API).
    Tuyệt đối KHÔNG có phép biến đổi ngẫu nhiên (No Random Augmentations).
    Mục tiêu (Designing ML Systems Ch. 5 - Tránh Data Leakage & Đảm bảo Tính Nhất Quán):
    - Resize ảnh về kích thước chuẩn định sẵn (256 rồi CenterCrop về image_size, hoặc Resize trực tiếp).
    - Chuẩn hóa ToTensor và Normalize với cùng ImageNet Mean/Std.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Tính toán Class Weights để đưa vào CrossEntropyLoss(weight=class_weights)
    nhằm giải quyết triệt để Class Imbalance (Chip Huyen - Ch. 4).
    Công thức cân bằng nghịch đảo tần suất:
        w_j = Total_Samples / (Num_Classes * Count_j)
    """
    targets = [s[1] for s in dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets))
    total_samples = len(targets)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts.float())
    return weights

def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = False,
    dataset_version: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Dict[int, str], Optional[torch.Tensor]]:
    """
    Factory function xây dựng DataLoaders chuẩn mực cho Train và Validation.
    """
    train_path = Path(data_dir) / "train"
    valid_path = Path(data_dir) / "valid"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục {train_path} hoặc {valid_path}")

    # Tự động chọn transform tối ưu:
    # Nếu là Dataset 1 (có data leakage), kích hoạt pipeline đặc trị chống học vẹt (Cutout, Tight Zoom, Color Shift)
    is_v1 = (dataset_version == "v1") or (Path(data_dir).name == "data" and "v2" not in str(data_dir))
    train_tf = get_dataset1_train_transforms(image_size=image_size) if is_v1 else get_train_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(
        root=str(train_path),
        transform=train_tf
    )
    valid_dataset = datasets.ImageFolder(
        root=str(valid_path),
        transform=get_eval_transforms(image_size=image_size)
    )

    idx_to_class = {idx: cls_name for cls_name, idx in train_dataset.class_to_idx.items()}

    # Tính class weights cho loss function
    class_weights = compute_class_weights(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle bắt buộc cho Train (tránh batch correlation)
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,         # Không shuffle cho Valid để đảm bảo reproducibility
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, valid_loader, idx_to_class, class_weights

if __name__ == "__main__":
    print("=== KIỂM THỬ PIPELINE DỮ LIỆU (SANITY CHECK) ===")
    train_loader, valid_loader, idx_to_class, class_weights = build_dataloaders(
        data_dir="data",
        batch_size=32,
        image_size=224,
        num_workers=2,
        pin_memory=False
    )

    print(f"Tổng số lớp: {len(idx_to_class)}")
    print(f"Số lượng batches trong Train: {len(train_loader)}")
    print(f"Số lượng batches trong Valid: {len(valid_loader)}")

    # Trích xuất 1 batch kiểm tra kích thước và phân phối
    images, labels = next(iter(train_loader))
    print(f"Shape của 1 batch ảnh: {images.shape} (Batch, Channel, Height, Width)")
    print(f"Shape của 1 batch nhãn: {labels.shape}")
    print(f"Dải giá trị Pixel sau chuẩn hóa: Min = {images.min():.2f}, Max = {images.max():.2f}")
    print(f"Shape của Class Weights tensor: {class_weights.shape}")
    print("=== DATA PIPELINE ĐÃ HOẠT ĐỘNG HOÀN HẢO! ===")
