"""
Model Inference Service
Tuân thủ nguyên tắc:
- Designing Machine Learning Systems (Chip Huyen) - Chương 7 (Model Serving & Latency)
- Hands-On Machine Learning (Aurélien Géron) - Chương 13, 14
"""

import io
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# Chuẩn hóa ImageNet đồng nhất tuyệt đối với quá trình training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model_architecture(model_name: str, num_classes: int) -> nn.Module:
    """Model Factory cho phép cắm rút (Plug & Play) nhiều kiến trúc PyTorch."""
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


class ModelService:
    def __init__(self, model_path: str = "models/baseline1_simple_cnn.pth", mapping_path: str = "models/class_indices.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path)
        self.model = None
        self.backend_type = "pytorch"  # "pytorch" hoặc "onnx"
        self.onnx_input_name = None
        self.idx_to_class: Dict[int, str] = {}
        self.model_version = self.model_path.stem

        # Deterministic preprocessing pipeline cho serving (Géron Ch. 13)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.load_metadata()
        self.load_model()

    def load_metadata(self):
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file nhãn tại {self.mapping_path}")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # JSON keys are strings, convert to int
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

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.model = session
            self.backend_type = "onnx"
            self.onnx_input_name = session.get_inputs()[0].name
            self.model_version = f"{self.model_path.stem} (ONNX)"
            print(f"[OK] Đã tải thành công mô hình ONNX từ {self.model_path} | providers={session.get_providers()}")
            return

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        num_classes = checkpoint.get("num_classes", len(self.idx_to_class))
        model_name = checkpoint.get("model_name", "simple_cnn")

        self.model = build_model_architecture(model_name, num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.backend_type = "pytorch"
        self.model_version = f"{model_name} ({self.model_path.stem})"
        print(f"[OK] Đã tải thành công mô hình '{model_name}' từ {self.model_path} trên thiết bị: {self.device}")

    def _preprocess_onnx(self, image: Image.Image) -> np.ndarray:
        img = image.resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        return np.transpose(arr, (2, 0, 1))[None, ...]  # (1, 3, 224, 224)

    def predict(self, image_bytes: bytes) -> Tuple[str, float, List[Dict[str, float]], float]:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.backend_type == "onnx":
            inp = self._preprocess_onnx(image)
            start_time = time.perf_counter()
            logits = self.model.run(None, {self.onnx_input_name: inp})[0][0]
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            # Numerically stable softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            k = min(3, len(probs))
            top3_indices = np.argsort(probs)[::-1][:k]
            top3_list = [
                {"class_name": self.idx_to_class.get(int(idx), f"Class_{int(idx)}"), "confidence": round(float(probs[idx]), 4)}
                for idx in top3_indices
            ]
        else:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            start_time = time.perf_counter()
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
            inference_time_ms = (time.perf_counter() - start_time) * 1000

            top3_probs, top3_indices = torch.topk(probs, k=min(3, len(probs)))
            top3_list = [
                {"class_name": self.idx_to_class.get(idx.item(), f"Class_{idx.item()}"), "confidence": round(prob.item(), 4)}
                for prob, idx in zip(top3_probs, top3_indices)
            ]

        best_class = top3_list[0]["class_name"]
        best_conf = top3_list[0]["confidence"]

        return best_class, best_conf, top3_list, round(inference_time_ms, 2)
