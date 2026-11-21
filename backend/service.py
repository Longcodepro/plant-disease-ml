"""
Model Inference Service
Tuân thủ nguyên tắc:
- Designing Machine Learning Systems (Chip Huyen) - Chương 7 (Serving) & Chương 8 (Monitoring)
- Hands-On Machine Learning (Aurélien Géron) - Chương 13, 14

Tính năng nổi bật:
- Lazy-import PyTorch/Torchvision: chạy `.onnx` không cần cài hay nạp PyTorch (~2 GB).
- Warmup lúc khởi động để loại bỏ cold-start.
- SHA-256 In-Memory LRU Cache: ảnh trùng lặp trả về trong <0.5 ms (`cache_hit=True`).
- Batched Inference (`predict_batch`) tận dụng `dynamic_axes` của ONNX.
- Explainable AI (`explain`): Batched Occlusion Sensitivity Heatmap thuần ONNX/NumPy/PIL.
- Online Telemetry (`get_metrics`): theo dõi p50/p95 latency, cache hit rate, tỷ lệ uncertain (data drift).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

# Chuẩn hóa ImageNet đồng nhất tuyệt đối với quá trình training
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
DEFAULT_CACHE_SIZE = int(os.getenv("PREDICT_CACHE_SIZE", "256"))


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


def _jet_colormap(norm_2d: np.ndarray) -> np.ndarray:
    """Ánh xạ bản đồ [0, 1] -> RGB Jet colormap thuần NumPy (không cần matplotlib/opencv)."""
    x = np.clip(norm_2d, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


class TelemetryTracker:
    """Bộ thu thập metrics trực tuyến an toàn luồng (Chip Huyen Ch. 8 - Monitoring)."""

    def __init__(self, window_size: int = 1000):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_images = 0
        self.cache_hits = 0
        self.uncertain_count = 0
        self.latencies_ms: deque[float] = deque(maxlen=window_size)
        self.class_counts: Counter[str] = Counter()

    def record(
        self,
        latency_ms: float,
        predicted_classes: List[str],
        uncertain_flags: List[bool],
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_images += len(predicted_classes)
            if cache_hit:
                self.cache_hits += 1
            self.uncertain_count += sum(1 for u in uncertain_flags if u)
            self.latencies_ms.append(float(latency_ms))
            self.class_counts.update(predicted_classes)

    def snapshot(self, model_version: str, backend_type: str) -> Dict[str, Any]:
        with self._lock:
            lats = list(self.latencies_ms)
            if lats:
                arr = np.array(lats, dtype=np.float64)
                lat_stats = {
                    "avg": round(float(np.mean(arr)), 2),
                    "p50": round(float(np.percentile(arr, 50)), 2),
                    "p95": round(float(np.percentile(arr, 95)), 2),
                    "min": round(float(np.min(arr)), 2),
                    "max": round(float(np.max(arr)), 2),
                }
            else:
                lat_stats = {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "model_version": model_version,
                "backend_type": backend_type,
                "total_requests": self.total_requests,
                "total_images_predicted": self.total_images,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": round(self.cache_hits / self.total_requests, 4) if self.total_requests else 0.0,
                "uncertain_count": self.uncertain_count,
                "uncertain_rate": round(self.uncertain_count / self.total_images, 4) if self.total_images else 0.0,
                "latency_ms": lat_stats,
                "top_predicted_classes": dict(self.class_counts.most_common(5)),
            }


class ModelService:
    def __init__(
        self,
        model_path: str = "models/convnext_tiny_khoi_int8.onnx",
        mapping_path: str = "models/class_indices.json",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ):
        self.model_path = Path(model_path)
        self.mapping_path = Path(mapping_path)
        self.confidence_threshold = confidence_threshold
        self.model: Any = None
        self.device: Any = None
        self.backend_type = "onnx" if self.model_path.suffix.lower() == ".onnx" else "pytorch"
        self.onnx_input_name: Optional[str] = None
        self._cam_session: Any = None
        self._cam_weights: Optional[np.ndarray] = None  # (C, num_classes) kết hợp LayerNorm gamma * W
        self.idx_to_class: Dict[int, str] = {}
        self.model_version = self.model_path.stem

        self._cache_size = max(0, cache_size)
        self._cache_lock = threading.Lock()
        self._predict_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.telemetry = TelemetryTracker()

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

            dummy = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            session.run(None, {self.onnx_input_name: dummy})
            self._init_onnx_cam(providers)
            print(f"[OK] Đã tải & warmup mô hình ONNX từ {self.model_path} | providers={session.get_providers()}")
            return

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

    def _decode_and_resize(self, image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise ValueError("File ảnh rỗng.")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"File không phải ảnh hợp lệ hoặc bị hỏng: {exc}") from exc
        return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    def _pil_to_chw(self, img_224: Image.Image) -> np.ndarray:
        arr = (np.asarray(img_224, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))

    def _preprocess_bytes(self, image_bytes: bytes) -> np.ndarray:
        return self._pil_to_chw(self._decode_and_resize(image_bytes))

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
        digest = hashlib.sha256(image_bytes).hexdigest() if self._cache_size > 0 else ""

        # 1. Kiểm tra SHA-256 LRU Cache
        if digest:
            t_cache = time.perf_counter()
            with self._cache_lock:
                cached = self._predict_cache.get(digest)
                if cached is not None:
                    self._predict_cache.move_to_end(digest)
                    hit_ms = round((time.perf_counter() - t_cache) * 1000.0, 2)
                    # Tính lại ngưỡng uncertain động theo confidence_threshold hiện tại
                    is_uncertain = cached["confidence"] < self.confidence_threshold
                    warning = (
                        f"Độ tin cậy thấp ({cached['confidence'] * 100:.1f}% < {self.confidence_threshold * 100:.0f}%). "
                        "Ảnh có thể bị mờ, thiếu sáng hoặc không phải lá cây thuộc 38 lớp đã huấn luyện."
                        if is_uncertain
                        else None
                    )
                    res = {
                        **cached,
                        "is_uncertain": is_uncertain,
                        "warning": warning,
                        "cache_hit": True,
                        "inference_time_ms": hit_ms,
                    }
                    self.telemetry.record(hit_ms, [res["predicted_class"]], [is_uncertain], cache_hit=True)
                    return res

        # 2. Chạy suy luận thật
        batch = self._preprocess_bytes(image_bytes)[None, ...]
        logits, inference_time_ms = self._run_batch_logits(batch)
        probs = _softmax_2d(logits)[0]
        base = self._format_prediction(probs)
        result = {
            **base,
            "cache_hit": False,
            "inference_time_ms": round(inference_time_ms, 2),
            "model_version": self.model_version,
        }

        if digest:
            with self._cache_lock:
                self._predict_cache[digest] = {
                    "predicted_class": base["predicted_class"],
                    "confidence": base["confidence"],
                    "top_3": base["top_3"],
                    "model_version": self.model_version,
                }
                self._predict_cache.move_to_end(digest)
                if len(self._predict_cache) > self._cache_size:
                    self._predict_cache.popitem(last=False)

        self.telemetry.record(
            result["inference_time_ms"],
            [result["predicted_class"]],
            [result["is_uncertain"]],
            cache_hit=False,
        )
        return result

    def predict_batch(self, image_bytes_list: List[bytes]) -> Tuple[List[Dict[str, Any]], float]:
        if not image_bytes_list:
            raise ValueError("Danh sách ảnh rỗng.")
        batch = np.stack([self._preprocess_bytes(b) for b in image_bytes_list], axis=0)
        logits, total_ms = self._run_batch_logits(batch)
        probs = _softmax_2d(logits)
        items = [self._format_prediction(probs[i]) for i in range(len(probs))]
        rounded_ms = round(total_ms, 2)
        self.telemetry.record(
            rounded_ms,
            [it["predicted_class"] for it in items],
            [it["is_uncertain"] for it in items],
            cache_hit=False,
        )
        return items, rounded_ms

    def _init_onnx_cam(self, providers: List[str]) -> None:
        """
        Khởi tạo nhánh Exact CAM (Class Activation Mapping — tương đương Grad-CAM cho mạng GAP)
        trực tiếp từ đồ thị ONNX (hỗ trợ cả FP32 & INT8) mà không cần PyTorch.
        """
        try:
            import onnx
            import onnxruntime as ort
            from onnx import TensorProto, helper, numpy_helper

            m = onnx.load(str(self.model_path))
            gap_node = next((n for n in m.graph.node if n.op_type == "GlobalAveragePool"), None)
            if gap_node is None:
                return

            inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

            # Lấy trọng số lớp Linear cuối (hỗ trợ cả INT8 quantized & FP32)
            W: Optional[np.ndarray] = None
            if "classifier.2.weight_Transposed_quantized" in inits:
                wq = inits["classifier.2.weight_Transposed_quantized"].astype(np.float32)
                ws = inits["classifier.2.weight_Transposed_scale"].astype(np.float32)
                wz = inits["classifier.2.weight_Transposed_zero_point"].astype(np.float32)
                W = (wq - wz) * ws  # (C, num_classes)
            elif "classifier.2.weight" in inits:
                W = inits["classifier.2.weight"].astype(np.float32).T
            elif "fc.weight" in inits:
                W = inits["fc.weight"].astype(np.float32).T

            if W is None:
                return

            # Nhân thêm gamma của LayerNorm (nếu có trong ConvNeXt head) để đúng gradient chính xác
            if "classifier.0.weight" in inits and inits["classifier.0.weight"].shape[0] == W.shape[0]:
                W = W * inits["classifier.0.weight"].astype(np.float32)[:, None]

            fmap_name = gap_node.input[0]
            m.graph.output.append(helper.make_tensor_value_info(fmap_name, TensorProto.FLOAT, None))
            self._cam_session = ort.InferenceSession(m.SerializeToString(), providers=providers)
            self._cam_weights = W
            print(f"[OK] Đã bật Exact ONNX CAM/Grad-CAM trên feature map '{fmap_name}'")
        except Exception as exc:
            print(f"[INFO] Không bật được ONNX CAM ({exc}); dùng Occlusion Sensitivity fallback.")
            self._cam_session = None
            self._cam_weights = None

    def _overlay_heatmap(self, img_224: Image.Image, map_2d: np.ndarray) -> str:
        """Chuẩn hóa bản đồ nhiệt mượt, cắt nhiễu nền và chồng màu Jet lên ảnh lá -> Base64 PNG."""
        lo, hi = float(map_2d.min()), float(map_2d.max())
        norm = (map_2d - lo) / (hi - lo + 1e-8) if hi - lo > 1e-8 else np.zeros_like(map_2d)

        smooth_pil = (
            Image.fromarray((norm * 255).astype(np.uint8))
            .resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
            .filter(ImageFilter.GaussianBlur(radius=6))
        )
        sn = np.asarray(smooth_pil, dtype=np.float32) / 255.0
        # Lọc nền yếu (< 0.15) để giữ sạch nền ảnh, chỉ tô nổi vùng đốm bệnh
        sn = np.clip((sn - 0.15) / 0.85, 0.0, 1.0)

        heat_rgb = _jet_colormap(sn).astype(np.float32)
        orig_rgb = np.asarray(img_224, dtype=np.float32)
        alpha = (0.62 * (sn ** 1.15))[..., None]
        overlay = np.clip((1.0 - alpha) * orig_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def explain(self, image_bytes: bytes, grid: int = 5, patch_size: int = 72) -> Dict[str, Any]:
        """
        Giải thích XAI:
        - Ưu tiên **Exact ONNX Class Activation Mapping (CAM / Grad-CAM)** trên feature map 7x7 cuối cùng
          (1 lần chạy duy nhất ~25 ms, khoanh chính xác vùng đốm bệnh trên lá, không cần PyTorch).
        - Tự động fallback sang Batched Occlusion Sensitivity nếu model không có nhánh CAM.
        """
        t0 = time.perf_counter()
        img_224 = self._decode_and_resize(image_bytes)
        base_chw = self._pil_to_chw(img_224)

        if self._cam_session is not None and self._cam_weights is not None:
            logits, fmap = self._cam_session.run(None, {self.onnx_input_name: base_chw[None, ...]})
            base_probs = _softmax_2d(np.asarray(logits, dtype=np.float32))[0]
            target_idx = int(np.argmax(base_probs))
            pred_info = self._format_prediction(base_probs)

            # fmap: (1, C, H, W) -> (H, W, C); CAM = ReLU(fmap @ (gamma * W[:, target]))
            f_hwc = np.transpose(fmap[0], (1, 2, 0))
            cam_map = np.maximum(0.0, f_hwc @ self._cam_weights[:, target_idx])
            b64 = self._overlay_heatmap(img_224, cam_map)
            return {
                **pred_info,
                "method": "Exact Class Activation Mapping (CAM / Grad-CAM, ONNX)",
                "heatmap_base64": b64,
                "explain_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "model_version": self.model_version,
            }

        # Fallback: Batched Occlusion Sensitivity
        base_logits, _ = self._run_batch_logits(base_chw[None, ...])
        base_probs = _softmax_2d(base_logits)[0]
        target_idx = int(np.argmax(base_probs))
        base_logit = float(base_logits[0, target_idx])
        pred_info = self._format_prediction(base_probs)

        coords = np.linspace(0, IMAGE_SIZE - patch_size, grid, dtype=int)
        occluded_batch = np.repeat(base_chw[None, ...], grid * grid, axis=0)
        boxes: List[Tuple[int, int, int, int]] = []
        idx = 0
        for y in coords:
            for x in coords:
                occluded_batch[idx, :, y : y + patch_size, x : x + patch_size] = 0.0
                boxes.append((int(y), int(y + patch_size), int(x), int(x + patch_size)))
                idx += 1

        occ_logits, _ = self._run_batch_logits(occluded_batch)
        drops = np.maximum(0.0, base_logit - occ_logits[:, target_idx])

        importance = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        counts = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for drop, (y1, y2, x1, x2) in zip(drops, boxes):
            importance[y1:y2, x1:x2] += float(drop)
            counts[y1:y2, x1:x2] += 1.0

        b64 = self._overlay_heatmap(img_224, importance / np.maximum(counts, 1.0))
        return {
            **pred_info,
            "method": f"Batched Occlusion Sensitivity ({grid}x{grid} patches)",
            "heatmap_base64": b64,
            "explain_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "model_version": self.model_version,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.telemetry.snapshot(self.model_version, self.backend_type)
