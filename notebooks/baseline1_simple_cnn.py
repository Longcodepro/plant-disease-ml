"""
Baseline 1: Vanilla CNN from Scratch
Dự án: Chẩn đoán bệnh trên lá cây nông nghiệp
Tuân thủ nguyên tắc:
- Designing Machine Learning Systems (Chip Huyen) - Chương 6 (Simple Baseline First) & Chương 7 (Model Size & Latency)
- Hands-On Machine Learning (Aurélien Géron) - Chương 14 (Deep Computer Vision with CNNs)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Import module pipeline đã xây dựng ở Bước 1
sys.path.append(str(Path(__file__).parent))
from data_pipeline import build_dataloaders


# ==============================================================================
# 1. KIẾN TRÚC MẠNG CNN ĐƠN GIẢN (VANILLA CNN)
# Tham khảo Géron Chương 14: Khối [Conv2d -> BatchNorm -> ReLU -> MaxPool2d] lặp lại
# Kết hợp AdaptiveAvgPool2d để kiểm soát chặt chẽ số lượng tham số (Model Footprint)
# ==============================================================================
class SimplePlantCNN(nn.Module):
    def __init__(self, num_classes: int = 38):
        super(SimplePlantCNN, self).__init__()

        # Feature Extractor: 4 khối tích chập trích xuất đặc trưng từ thấp (cạnh lá, vân) đến cao (đốm bệnh, mốc)
        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Pooling: Giảm chiều không gian về 1x1, tránh bùng nổ tham số so với Flatten trực tiếp
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head: Đưa ra xác suất dự đoán cho 38 lớp
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),  # Regularization chống Overfitting
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


# ==============================================================================
# 2. HÀM TÍNH TOÁN METRICS VÀ ĐO ĐẠC HỆ THỐNG (SYSTEM EVALUATION)
# Tham khảo Chip Huyen Chương 4 & 7: Macro F1, Latency và Model Footprint
# ==============================================================================
def calculate_model_footprint(model: nn.Module) -> Tuple[int, float]:
    """Tính toán số lượng tham số và dung lượng ước tính của mô hình (MB)."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4 bytes cho mỗi tham số float32
    size_mb = (num_params * 4) / (1024 * 1024)
    return num_params, size_mb


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Huấn luyện 1 epoch và trả về Loss trung bình, Accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Đánh giá mô hình trên tập Validation:
    - ML Metrics: Loss, Accuracy, Macro F1-Score, Weighted F1-Score
    - System Metrics: Average Inference Latency per image (ms)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            start_t = time.perf_counter()
            outputs = model(images)
            end_t = time.perf_counter()

            # Đo độ trễ inference trung bình cho mỗi ảnh
            batch_latency = ((end_t - start_t) / images.size(0)) * 1000  # ms
            latencies.append(batch_latency)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = running_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    avg_latency = float(np.mean(latencies))

    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_macro_f1": macro_f1,
        "val_weighted_f1": weighted_f1,
        "avg_latency_ms": avg_latency
    }


# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (TRAINING & BENCHMARKING ENGINE)
# ==============================================================================
def run_baseline1(
    data_dir: str = "data",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    subset_fraction: float = 1.0,
    save_path: str = "models/baseline1_simple_cnn.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f" KHỞI ĐỘNG BASELINE 1: VANILLA CNN (FROM SCRATCH) ")
    print(f" Thiết bị tính toán: {device}")
    print(f"==================================================")

    # 1. Tải DataLoaders qua Data Pipeline đã xây dựng
    train_loader, valid_loader, idx_to_class, class_weights = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=224,
        num_workers=2,
        pin_memory=False
    )

    # Hỗ trợ chạy subset nhanh để kiểm thử (Dry-run / Quick Benchmark)
    if subset_fraction < 1.0:
        print(f"[*] Sử dụng {subset_fraction*100:.1f}% dữ liệu để kiểm thử nhanh vòng lặp huấn luyện...")
        train_sub_indices = list(range(int(len(train_loader.dataset) * subset_fraction)))
        valid_sub_indices = list(range(int(len(valid_loader.dataset) * subset_fraction)))
        train_loader = DataLoader(
            Subset(train_loader.dataset, train_sub_indices),
            batch_size=batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            Subset(valid_loader.dataset, valid_sub_indices),
            batch_size=batch_size,
            shuffle=False
        )

    # 2. Khởi tạo Mô hình & Kiểm tra Model Footprint (Chip Huyen Ch. 7)
    model = SimplePlantCNN(num_classes=len(idx_to_class)).to(device)
    num_params, size_mb = calculate_model_footprint(model)
    print(f"[*] Tổng số tham số (Parameters): {num_params:,}")
    print(f"[*] Kích thước mô hình ước tính: {size_mb:.2f} MB")

    # 3. Loss function với Class Weights & Optimizer Adam
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Training Loop
    best_macro_f1 = 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print("\n--- BẮT ĐẦU HUẤN LUYỆN & THEO DÕI METRICS ---")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, valid_loader, criterion, device)
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} - "
            f"Val Acc: {val_metrics['val_acc']*100:.2f}% - "
            f"Val Macro-F1: {val_metrics['val_macro_f1']:.4f} | "
            f"Latency: {val_metrics['avg_latency_ms']:.2f} ms/ảnh"
        )

        # Lưu checkpoint nếu đạt F1-Score cao nhất
        if val_metrics["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["val_macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": len(idx_to_class),
                "best_macro_f1": best_macro_f1,
                "metrics": val_metrics
            }, save_path)
            print(f"  --> Đã lưu Best Checkpoint tại: {save_path} (Macro-F1: {best_macro_f1:.4f})")

    # Kiểm tra kích thước file checkpoint thực tế trên đĩa cứng
    actual_file_size = Path(save_path).stat().st_size / (1024 * 1024) if Path(save_path).exists() else 0
    print("\n==================================================")
    print(" KẾT QUẢ TỔNG KẾT BASELINE 1 (BENCHMARK BASELINE)")
    print(f" 1. Best Macro F1-Score : {best_macro_f1:.4f}")
    print(f" 2. Model Binary Size   : {actual_file_size:.2f} MB")
    print(f" 3. Average Latency     : {val_metrics['avg_latency_ms']:.2f} ms/ảnh")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline 1: Vanilla CNN")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học (Learning Rate)")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Tỷ lệ tập dữ liệu sử dụng (0.01 đến 1.0)")
    parser.add_argument("--save_path", type=str, default="models/baseline1_simple_cnn.pth", help="Đường dẫn lưu trọng số")
    args = parser.parse_args()

    run_baseline1(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_fraction=args.subset_fraction,
        save_path=args.save_path
    )
