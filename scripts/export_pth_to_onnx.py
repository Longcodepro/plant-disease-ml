#!/usr/bin/env python3
"""
Xuất checkpoint PyTorch (.pth) sang ONNX và ONNX INT8 mà không cần huấn luyện lại.
Tác giả: Long
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi .pth sang .onnx và .onnx INT8")
    parser.add_argument("--pth", type=str, default="models/resnet18_long_v3_run_01.pth",
                        help="Đường dẫn file .pth")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Lượng tử hóa sang INT8")
    args = parser.parse_args()

    pth_path = Path(args.pth)
    if not pth_path.is_absolute():
        pth_path = ROOT_DIR / pth_path

    if not pth_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {pth_path}")

    print(f"📦 Đang nạp mô hình từ: {pth_path.name}...")
    checkpoint = torch.load(pth_path, map_location="cpu")
    num_classes = checkpoint.get("num_classes", 38)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Khởi tạo kiến trúc ResNet-18
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Tên file ONNX
    base_name = pth_path.stem
    onnx_path = pth_path.parent / f"{base_name}.onnx"
    onnx_int8_path = pth_path.parent / f"{base_name}_int8.onnx"

    # 1. Xuất ONNX FP32
    print(f"🌐 Đang xuất ONNX -> {onnx_path.name}...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    print(f"✅ Đã xuất ONNX thành công: {onnx_path.name} ({onnx_path.stat().st_size / 1024**2:.2f} MB)")

    # 2. Lượng tử hóa sang INT8
    if args.quantize:
        print(f"⚡ Đang lượng tử hóa INT8 -> {onnx_int8_path.name}...")
        try:
            from scripts.quantize_onnx import quantize_model
            quantize_model(onnx_path, onnx_int8_path)
            print(f"✅ Đã xuất INT8 thành công: {onnx_int8_path.name} ({onnx_int8_path.stat().st_size / 1024**2:.2f} MB)")
        except Exception as e:
            print(f"⚠️ Chưa thể tạo INT8: {e}")

    print(f"\n🎉 Hoàn thành chuyển đổi định dạng xuất sắc!")

if __name__ == "__main__":
    main()
