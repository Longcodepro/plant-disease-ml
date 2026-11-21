#!/usr/bin/env python3
"""
Nén mô hình ONNX từ FP32 sang INT8 (hỗ trợ cả Dynamic & Static Quantization) & đo benchmark so sánh.

Hai chế độ nén:
1. `--mode dynamic` (Dynamic Quantization - QUInt8 trên MatMul/Gemm):
   - Không cần dữ liệu calibration.
   - Giảm dung lượng ~3.3x (ConvNeXt-Tiny: 106.3 MB -> 32.3 MB), giữ nguyên độ chính xác (Cosine Sim 0.9992).
   - Lưu ý: Trên CPU không có AVX-512 VNNI (vd: CPU ảo Colab), có thể chậm hơn FP32 do overhead
     `DynamicQuantizeLinear` lúc chạy. Phù hợp khi ưu tiên giảm dung lượng lưu trữ/Docker/Git (<50 MB).

2. `--mode static` (Static Post-Training Quantization - QDQ + Calibration Data):
   - Dùng ~100-200 ảnh mẫu từ tập train để tính cố định scale/zero-point của activation trước.
   - Loại bỏ bước `DynamicQuantizeLinear` lúc runtime => vừa nhẹ vừa tối ưu tốc độ trên CPU hỗ trợ INT8.

Cách chạy:
    # Nén Dynamic INT8:
    python scripts/quantize_onnx.py --mode dynamic \
        --input models/convnext_tiny_khoi.onnx \
        --output models/convnext_tiny_khoi_int8.onnx

    # Nén Static INT8 (cần thư mục data/train để calibrate):
    python scripts/quantize_onnx.py --mode static \
        --input models/convnext_tiny_khoi.onnx \
        --output models/convnext_tiny_khoi_static_int8.onnx \
        --data-dir data --calib-samples 150 --eval-valid --max-samples 1600
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


def _preprocess_for_quant(input_path: Path, prep_path: Path) -> Path:
    """Chạy ONNX shape inference & tối ưu đồ thị trước khi lượng tử hóa (khuyến nghị của ORT)."""
    try:
        quant_pre_process(str(input_path), str(prep_path), skip_symbolic_shape=False)
        return prep_path
    except Exception as exc:
        print(f"⚠️  quant_pre_process bỏ qua ({exc}); tiếp tục với đồ thị gốc.")
        return input_path


class ImageFolderCalibrationReader(CalibrationDataReader):
    """Đọc ảnh mẫu rải đều 38 lớp từ data/train để calibrate cho Static INT8 Quantization."""

    def __init__(self, data_dir: str, input_name: str, image_size: int = 224, num_samples: int = 150):
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from torchvision import datasets
        from notebooks.data_pipeline import get_eval_transforms

        train_dir = Path(data_dir) / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Static quantization cần thư mục calibration: {train_dir}")

        ds = datasets.ImageFolder(str(train_dir), transform=get_eval_transforms(image_size))
        # Lấy mẫu rải đều theo bước nhảy để phủ đều cả 38 lớp
        step = max(1, len(ds) // num_samples)
        indices = list(range(0, len(ds), step))[:num_samples]
        self._input_name = input_name
        self._samples = [ds[i][0].unsqueeze(0).numpy().astype(np.float32) for i in indices]
        self._iter = iter(self._samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._iter, None)
        return None if batch is None else {self._input_name: batch}

    def rewind(self) -> None:
        self._iter = iter(self._samples)


def quantize_model(
    input_path: Path,
    output_path: Path,
    mode: str = "dynamic",
    data_dir: str = "data",
    image_size: int = 224,
    calib_samples: int = 150,
) -> None:
    """Thực hiện Dynamic hoặc Static INT8 Quantization và lưu ra output_path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ONNX đầu vào: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        prepped = _preprocess_for_quant(input_path, Path(tmp_dir) / "prepped.onnx")

        if mode == "dynamic":
            # ~95% tham số ConvNeXt nằm ở nn.Linear (MatMul/Gemm); dùng QUInt8 tránh nút thắt ConvInteger
            quantize_dynamic(
                model_input=str(prepped),
                model_output=str(output_path),
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
        elif mode == "static":
            sess = ort.InferenceSession(str(prepped), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            reader = ImageFolderCalibrationReader(
                data_dir=data_dir,
                input_name=input_name,
                image_size=image_size,
                num_samples=calib_samples,
            )
            quantize_static(
                model_input=str(prepped),
                model_output=str(output_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.MinMax,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            )
        else:
            raise ValueError(f"mode phải là 'dynamic' hoặc 'static', nhận '{mode}'")


def benchmark_session(model_path: Path, dummy: np.ndarray, warmup: int = 5, runs: int = 30) -> tuple[np.ndarray, float]:
    """Đo CPU latency trung bình 1 ảnh (ms) và trả về logits mẫu để kiểm tra sai số."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    for _ in range(warmup):
        out = sess.run(None, {input_name: dummy})[0]

    t0 = time.perf_counter()
    for _ in range(runs):
        out = sess.run(None, {input_name: dummy})[0]
    avg_ms = (time.perf_counter() - t0) / runs * 1000.0
    return out, avg_ms


def evaluate_onnx_on_valid(
    model_path: Path,
    data_dir: str,
    batch_size: int = 32,
    max_samples: int = 0,
    max_batches: int = 0,
) -> dict:
    """
    Đánh giá Accuracy & Macro-F1 của file ONNX trên tập valid.
    Khi lấy tập con (`max_samples > 0` hoặc `max_batches > 0`), lấy mẫu rải đều (strided)
    để phủ đủ toàn bộ 38 lớp thay vì chỉ lấy các lớp đầu tiên theo thứ tự alphabet.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from notebooks.data_pipeline import get_eval_transforms

    valid_ds = datasets.ImageFolder(str(Path(data_dir) / "valid"), transform=get_eval_transforms(224))
    limit = max_samples if max_samples > 0 else (max_batches * batch_size if max_batches > 0 else 0)
    if 0 < limit < len(valid_ds):
        step = max(1, len(valid_ds) // limit)
        indices = list(range(0, len(valid_ds), step))[:limit]
        valid_ds = Subset(valid_ds, indices)

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    all_preds, all_targets = [], []
    for images, targets in valid_loader:
        logits = sess.run(None, {input_name: images.numpy()})[0]
        all_preds.extend(np.argmax(logits, axis=1).tolist())
        all_targets.extend(targets.numpy().tolist())

    return {
        "samples": len(all_targets),
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "macro_f1": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Nén ONNX FP32 -> INT8 (Dynamic / Static Quantization)")
    parser.add_argument("--mode", choices=("dynamic", "static"), default="dynamic", help="Chế độ nén INT8")
    parser.add_argument("--input", "-i", type=str, default="models/convnext_tiny_khoi.onnx", help="File ONNX FP32 gốc")
    parser.add_argument("--output", "-o", type=str, default=None, help="File ONNX INT8 đầu ra")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--calib-samples", type=int, default=150, help="Số ảnh calibration cho chế độ static")
    parser.add_argument("--runs", type=int, default=30, help="Số lần chạy đo latency")
    parser.add_argument("--eval-valid", action="store_true", help="Đánh giá thêm Accuracy/Macro-F1 trên data/valid")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=0, help="Số ảnh valid rải đều 38 lớp (0 = toàn bộ)")
    parser.add_argument("--max-batches", type=int, default=0, help="Tương thích ngược (0 = toàn bộ)")
    args = parser.parse_args()

    in_path = Path(args.input)
    suffix = "_static_int8.onnx" if args.mode == "static" else "_int8.onnx"
    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}{suffix}")

    print(f"🗜️  Đang nén INT8 ({args.mode}): {in_path} -> {out_path}")
    quantize_model(
        in_path,
        out_path,
        mode=args.mode,
        data_dir=args.data_dir,
        image_size=args.image_size,
        calib_samples=args.calib_samples,
    )

    fp32_mb = in_path.stat().st_size / (1024 ** 2)
    int8_mb = out_path.stat().st_size / (1024 ** 2)

    rng = np.random.default_rng(42)
    dummy = rng.standard_normal((1, 3, args.image_size, args.image_size), dtype=np.float32)

    fp32_out, fp32_ms = benchmark_session(in_path, dummy, runs=args.runs)
    int8_out, int8_ms = benchmark_session(out_path, dummy, runs=args.runs)

    a, b = fp32_out.reshape(-1), int8_out.reshape(-1)
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print("\n============================================================")
    print(f" 📊 SO SÁNH ONNX FP32 vs ONNX INT8 [{args.mode.upper()}] (CPU)")
    print("============================================================")
    print(f" 📦 Dung lượng : {fp32_mb:.2f} MB  ->  {int8_mb:.2f} MB  (giảm {fp32_mb / int8_mb:.2f}x)")
    print(f" ⚡ Latency    : {fp32_ms:.1f} ms  ->  {int8_ms:.1f} ms  (tỷ lệ {fp32_ms / int8_ms:.2f}x)")
    print(f" 🎯 Cosine Sim : {cos_sim:.6f} (top-1 match: {int(a.argmax()) == int(b.argmax())})")
    print("============================================================\n")

    if args.eval_valid:
        print("🔎 Đang đánh giá độ chính xác thực tế trên tập Validation (rải đều 38 lớp)...")
        m_fp32 = evaluate_onnx_on_valid(in_path, args.data_dir, max_samples=args.max_samples, max_batches=args.max_batches)
        m_int8 = evaluate_onnx_on_valid(out_path, args.data_dir, max_samples=args.max_samples, max_batches=args.max_batches)
        print(f" FP32 ({m_fp32['samples']} ảnh): Acc = {m_fp32['accuracy']*100:.2f}% | Macro-F1 = {m_fp32['macro_f1']*100:.2f}%")
        print(f" INT8 ({m_int8['samples']} ảnh): Acc = {m_int8['accuracy']*100:.2f}% | Macro-F1 = {m_int8['macro_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
