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
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Chuẩn hóa ImageNet đồng nhất tuyệt đối với quá trình training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ModelService:
    def __init__(self, model_path: str = "models/baseline1_simple_cnn.pth", mapping_path: str = "models/class_indices.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path)
        self.model = None
        self.idx_to_class: Dict[int, str] = {}
        self.model_version = "baseline1_simple_cnn"

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

        from notebooks.baseline1_simple_cnn import SimplePlantCNN
        checkpoint = torch.load(self.model_path, map_location=self.device)
        num_classes = checkpoint.get("num_classes", len(self.idx_to_class))

        self.model = SimplePlantCNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[OK] Đã tải thành công mô hình từ {self.model_path} trên thiết bị: {self.device}")

    def predict(self, image_bytes: bytes) -> Tuple[str, float, List[Dict[str, float]], float]:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
