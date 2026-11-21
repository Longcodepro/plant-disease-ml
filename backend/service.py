"""
Model Inference Service
Tuân thủ nguyên tắc:
- Designing Machine Learning Systems (Chip Huyen) - Chương 7 (Model Serving & Latency)
- Hands-On Machine Learning (Aurélien Géron) - Chương 13, 14

Tối ưu triển khai:
- Lazy-import PyTorch/Torchvision: khi chạy model `.onnx` không cần cài hay nạp PyTorch (~2 GB).
- Warmup lúc khởi động để loại bỏ độ trễ cold-start ở request đầu tiên.
- Hỗ trợ suy luận theo lô (batched inference) tận dụng `dynamic_axes` của ONNX.
- Cảnh báo độ tin cậy thấp (`is_uncertain`) khi ảnh mờ hoặc ngoài phân phối 38 lớp.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# Chuẩn hóa ImageNet đồng nhất tuyệt đối với quá trình training
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))


def build_model_architecture(model_name: str, num_classes: int):
    """Model Factory cho phép cắm rút (Plug & Play) nhiều kiến trúc PyTorch (lazy import)."""
    import torch.nn as nn
    from torchvision import models

    name = model_name.lower().strip()

    if name in ("simple_cnn", "baseline1_simple_cnn"):
        from notebooks.baseline1_simple_cnn import SimplePlantCNN
        return SimplePlantCNN(num_classes=num_classes)

    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    if name == "convnext_small":
        m = models.convnext_small(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m

    if name == "swin_t":
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m

    raise ValueError(
        f"Kiến trúc '{model_name}' chưa được đăng ký trong build_model_architecture()."
    )


def _softmax_2d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


class ModelService:
    def __init__(
        self,
        model_path: str = "models/convnext_tiny_khoi_int8.onnx",
        mapping_path: str = "models/class_indices.json",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path)
        self.confidence_threshold = confidence_threshold
        self.model: Any = None
        self.device: Any = None
        self.backend_type = "onnx" if self.model_path.suffix.lower() == ".onnx" else "pytorch"
        self.onnx_input_name: Optional[str] = None
        self.idx_to_class: Dict[int, str] = {}
        self.model_version = self.model_path.stem

        self.load_metadata()
        self.load_model()

    def load_metadata(self):
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file nhãn tại {self.mapping_path}")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}

    def load_model(self):
        if not self.model_path.exists():
            print(f"[CẢNH BÁO] Chưa tìm thấy checkpoint tại {self.model_path}. Model chưa sẵn sàng.")
            return

        if self.model_path.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise RuntimeError("Cần cài 'onnxruntime' (pip install onnxruntime) để chạy model .onnx") from e

            avail = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.model = session
            self.backend_type = "onnx"
            self.onnx_input_name = session.get_inputs()[0].name
            self.model_version = f"{self.model_path.stem} (ONNX)"

            # Warmup 1 lần để request đầu tiên không bị trễ cold-start
            dummy = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            session.run(None, {self.onnx_input_name: dummy})
            print(f"[OK] Đã tải & warmup mô hình ONNX từ {self.model_path} | providers={session.get_providers()}")
            return

        # Nhánh PyTorch (.pth): chỉ import torch khi thật sự dùng checkpoint PyTorch
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        num_classes = checkpoint.get("num_classes", len(self.idx_to_class))
        model_name = checkpoint.get("model_name", "simple_cnn")

        self.model = build_model_architecture(model_name, num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.backend_type = "pytorch"
        self.model_version = f"{model_name} ({self.model_path.stem})"

        with torch.no_grad():
            dummy_t = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)
            self.model(dummy_t)
        print(f"[OK] Đã tải & warmup mô hình '{model_name}' từ {self.model_path} trên thiết bị: {self.device}")

    def _preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decode -> RGB -> Resize 224x224 (Bilinear) -> CHW Float32 chuẩn ImageNet."""
        if not image_bytes:
            raise ValueError("File ảnh rỗng.")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"File không phải ảnh hợp lệ hoặc bị hỏng: {exc}") from exc

        img = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))  # (3, 224, 224)

    def _run_batch_logits(self, batch_chw: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp.")

        if self.backend_type == "onnx":
            start = time.perf_counter()
            logits = self.model.run(None, {self.onnx_input_name: batch_chw})[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return np.asarray(logits, dtype=np.float32), elapsed_ms

        import torch

        tensor = torch.from_numpy(batch_chw).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor).cpu().numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return logits, elapsed_ms

    def _format_prediction(self, probs_1d: np.ndarray) -> Dict[str, Any]:
        k = min(3, len(probs_1d))
        top3_indices = np.argsort(probs_1d)[::-1][:k]
        top3_list = [
            {
                "class_name": self.idx_to_class.get(int(idx), f"Class_{int(idx)}"),
                "confidence": round(float(probs_1d[idx]), 4),
            }
            for idx in top3_indices
        ]
        best_class = top3_list[0]["class_name"]
        best_conf = top3_list[0]["confidence"]
        is_uncertain = best_conf < self.confidence_threshold
        warning = (
            f"Độ tin cậy thấp ({best_conf * 100:.1f}% < {self.confidence_threshold * 100:.0f}%). "
            "Ảnh có thể bị mờ, thiếu sáng hoặc không phải lá cây thuộc 38 lớp đã huấn luyện."
            if is_uncertain
            else None
        )
        return {
            "predicted_class": best_class,
            "confidence": best_conf,
            "top_3": top3_list,
            "is_uncertain": is_uncertain,
            "warning": warning,
        }

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        batch = self._preprocess_bytes(image_bytes)[None, ...]
        logits, inference_time_ms = self._run_batch_logits(batch)
        probs = _softmax_2d(logits)[0]
        result = self._format_prediction(probs)
        result["inference_time_ms"] = round(inference_time_ms, 2)
        result["model_version"] = self.model_version
        return result

    def predict_batch(self, image_bytes_list: List[bytes]) -> Tuple[List[Dict[str, Any]], float]:
        if not image_bytes_list:
            raise ValueError("Danh sách ảnh rỗng.")
        batch = np.stack([self._preprocess_bytes(b) for b in image_bytes_list], axis=0)
        logits, total_ms = self._run_batch_logits(batch)
        probs = _softmax_2d(logits)
        items = [self._format_prediction(probs[i]) for i in range(len(probs))]
        return items, round(total_ms, 2)
