"""
Module Đánh giá Mô hình Chuẩn dùng chung (Shared Evaluation Engine)
Dự án: Chẩn đoán bệnh trên lá cây nông nghiệp (Plant Disease Classification)

Tuân thủ nguyên tắc:
- Designing Machine Learning Systems (Chip Huyen) - Chương 4 & 7 (Metrics, Latency, Footprint)
- Hands-On Machine Learning (Aurélien Géron) - Chương 3 & 14 (Classification Metrics & Confusion Matrix)
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
import numpy as np


def calculate_model_footprint(model: nn.Module) -> Dict[str, Any]:
    """
    Đo lường độ lớn và tài nguyên mô hình (Model Footprint).
    Trả về số tham số và dung lượng ước tính trên RAM/Disk (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024)  # Float32 = 4 bytes

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(model_size_mb, 2)
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    HÀM ĐÁNH GIÁ CHUẨN DÙNG CHUNG CHO TOÀN BỘ CÁC MÔ HÌNH TRONG NHÓM.

    Bộ tiêu chí chuẩn:
    1. ML Metrics:
       - Accuracy (Độ chính xác tổng quan)
       - Macro F1-Score (Tiêu chí then chốt nhất cho 38 lớp, không bị thiên vị lớp nhiều ảnh)
       - Weighted F1-Score (F1 có trọng số theo số lượng mẫu)
       - Macro Precision & Macro Recall
    2. System Metrics:
       - Average Inference Latency per image (ms) (Độ trễ suy luận mỗi bức ảnh)
       - Throughput (Số ảnh xử lý được mỗi giây)
    3. Model Footprint:
       - Số lượng parameters & dung lượng mô hình (MB)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    latencies_per_image = []
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            # Đo độ trễ suy luận (Latency)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()

            # Độ trễ trung bình mỗi ảnh trong batch (mili-giây)
            latency_ms = ((end_time - start_time) / batch_size) * 1000
            latencies_per_image.append(latency_ms)

            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Tính toán các chỉ số
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    avg_latency = float(np.mean(latencies_per_image))
    throughput = (1000 / avg_latency) if avg_latency > 0 else 0.0

    footprint = calculate_model_footprint(model)

    result = {
        # 1. Thước đo Machine Learning
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),

        # 2. Thước đo Hệ thống phục vụ API (System Latency)
        "latency_per_image_ms": round(avg_latency, 2),
        "throughput_fps": round(throughput, 1),

        # 3. Kích thước mô hình
        "total_parameters": footprint["total_parameters"],
        "model_size_mb": footprint["model_size_mb"]
    }

    if criterion is not None and len(all_targets) > 0:
        result["val_loss"] = round(float(running_loss / len(all_targets)), 4)

    return result


def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Model"):
    """In bảng tóm tắt kết quả đánh giá đẹp mắt, dễ so sánh."""
    print(f"\n=======================================================")
    print(f" 📊 BẢNG ĐÁNH GIÁ CHUẨN: {model_name.upper()}")
    print(f"=======================================================")
    print(f" 🎯 Macro F1-Score (Tiêu chí chính) : {metrics['macro_f1'] * 100:.2f}%")
    print(f" 📈 Accuracy                         : {metrics['accuracy'] * 100:.2f}%")
    print(f" ⚖️  Weighted F1-Score               : {metrics['weighted_f1'] * 100:.2f}%")
    print(f" 🔍 Macro Precision / Recall         : {metrics['macro_precision'] * 100:.2f}% / {metrics['macro_recall'] * 100:.2f}%")
    print(f"-------------------------------------------------------")
    print(f" ⚡ Độ trễ suy luận (Latency)        : {metrics['latency_per_image_ms']} ms/ảnh")
    print(f" 🚀 Tốc độ xử lý (Throughput)        : {metrics['throughput_fps']} khung hình/giây (FPS)")
    print(f" 📦 Dung lượng mô hình               : {metrics['model_size_mb']} MB ({metrics['total_parameters']:,} tham số)")
    print(f"=======================================================\n")
