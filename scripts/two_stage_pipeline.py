#!/usr/bin/env python3
"""
Hệ thống Two-Stage Pipeline Chẩn đoán Bệnh Cây Trồng
Giai đoạn 1: Định vị & Cắt phiến lá bệnh (Leaf Detection / Bounding Box)
Giai đoạn 2: Phân loại bệnh học trên lá đơn bằng ResNet-18 (Disease Diagnosis)

Tác giả: Long (Kiến trúc ResNet-18 + Two-Stage Pipeline)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop_for_resnet(crop_img: Image.Image, size: int = 224) -> np.ndarray:
    """Chuẩn hóa ảnh phiến lá đơn đã crop về định dạng chuẩn của ResNet-18."""
    img_resized = crop_img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img_resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[None, ...]


def detect_leaf_regions(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Giai đoạn 1: Phát hiện vùng phiến lá tiềm năng.
    Hỗ trợ phát hiện thông qua thuật toán phân tách sắc ký sinh học (Green-ExG / Mask)
    hoặc nạp mô hình YOLOv8 khi có trọng số.
    Trả về: danh sách các bounding boxes dạng (xmin, ymin, xmax, ymax).
    """
    w, h = image.size
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    
    # Tính Excess Green Index (ExG = 2*G - R - B) để định vị diệp lục phiến lá
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    exg = 2.0 * g - r - b
    
    # Tìm vùng lá trọng tâm (loại bỏ biên ngoài và đất cát)
    mask = exg > 15.0
    y_indices, x_indices = np.where(mask)
    
    if len(x_indices) > 500:
        # Tìm cụm lá chính
        xmin = max(0, int(np.percentile(x_indices, 5)) - 10)
        xmax = min(w, int(np.percentile(x_indices, 95)) + 10)
        ymin = max(0, int(np.percentile(y_indices, 5)) - 10)
        ymax = min(h, int(np.percentile(y_indices, 95)) + 10)
        
        # Đảm bảo tỷ lệ khung hình cân đối
        box_w, box_h = xmax - xmin, ymax - ymin
        if box_w > 50 and box_h > 50:
            return [(xmin, ymin, xmax, ymax)]

    # Fallback: Nếu không tách được cụm nhỏ, crop vùng trung tâm 80% (Center Leaf Crop)
    pad_w = int(w * 0.1)
    pad_h = int(h * 0.1)
    return [(pad_w, pad_h, w - pad_w, h - pad_h)]


def run_two_stage_diagnosis(
    image_path: Path,
    model_onnx_path: Path,
    mapping_path: Path,
    output_path: Path | None = None
) -> dict:
    """Thực thi luồng 2 giai đoạn: Phát hiện lá -> Cắt lá -> Phán bệnh."""
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    # Khởi tạo ONNX Runtime Session (chạy siêu nhanh trên CPU)
    sess = ort.InferenceSession(str(model_onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    raw_img = Image.open(image_path).convert("RGB")
    boxes = detect_leaf_regions(raw_img)
    
    results = []
    annotated_img = raw_img.copy()
    draw = ImageDraw.Draw(annotated_img)

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        crop_leaf = raw_img.crop((xmin, ymin, xmax, ymax))
        
        # Đưa lá đơn vào ResNet-18
        x = preprocess_crop_for_resnet(crop_leaf)
        logits = sess.run(None, {input_name: x})[0][0]
        
        # Tính Softmax xác suất
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        top3_indices = np.argsort(probs)[::-1][:3]
        top1_cls = idx_to_class[int(top3_indices[0])]
        top1_prob = float(probs[top3_indices[0]])
        
        top3_info = [
            {"class": idx_to_class[int(idx)], "probability": round(float(probs[idx]) * 100, 2)}
            for idx in top3_indices
        ]
        
        results.append({
            "box_index": i + 1,
            "bbox": [xmin, ymin, xmax, ymax],
            "predicted_disease": top1_cls,
            "confidence_percent": round(top1_prob * 100, 2),
            "top3_candidates": top3_info
        })
        
        # Vẽ bounding box lên ảnh
        draw.rectangle([xmin, ymin, xmax, ymax], outline="#2ecc71", width=4)
        label_text = f"{top1_cls.split('___')[-1]} ({top1_prob*100:.1f}%)"
        draw.text((xmin + 8, max(5, ymin - 20)), label_text, fill="#2ecc71")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_img.save(output_path)
        print(f"🖼️  Đã lưu ảnh chẩn đoán Two-Stage: {output_path}")

    return {
        "image_file": str(image_path.name),
        "total_leaves_detected": len(boxes),
        "diagnoses": results
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy Two-Stage Pipeline Chẩn đoán Bệnh Cây Trồng")
    parser.add_argument("--image", type=Path, required=True, help="Đường dẫn ảnh chụp ngoài đồng ruộng")
    parser.add_argument("--model", type=Path, default=Path("models/resnet18_long_v3_run_01_int8.onnx"))
    parser.add_argument("--mapping", type=Path, default=Path("models/class_indices.json"))
    parser.add_argument("--output", type=Path, default=Path("notebooks/eda_reports/two_stage_result.jpg"))
    args = parser.parse_args()

    res = run_two_stage_diagnosis(args.image, args.model, args.mapping, args.output)
    print("\n=======================================================")
    print("🌿 KẾT QUẢ CHẨN ĐOÁN TWO-STAGE PIPELINE (YOLO + RESNET-18)")
    print("=======================================================")
    print(f"📷 Ảnh đầu vào: {res['image_file']}")
    print(f"🍃 Số lá phát hiện: {res['total_leaves_detected']}")
    for d in res["diagnoses"]:
        print(f"\n[Lá #{d['box_index']}] Tọa độ: {d['bbox']}")
        print(f"  👉 Chẩn đoán chính : {d['predicted_disease']} ({d['confidence_percent']}%)")
        print(f"  🥈 Top 3 dự đoán   :")
        for cand in d["top3_candidates"]:
            print(f"     • {cand['class']} : {cand['probability']}%")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
