#!/usr/bin/env python3
"""
Tách phiến lá từ tập dữ liệu thực tế PlantDoc bằng mô hình YOLOv8 Leaf Detector.
Tạo tập dữ liệu lá đơn sạch: data_plantdoc_cropped/test

Cách chạy:
    /home/long/miniconda3/envs/MachineLearning/bin/python scripts/crop_plantdoc_leaves.py
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent


def crop_plantdoc_test(
    src_dir: Path,
    out_dir: Path,
    yolo_model_path: Path,
    conf: float = 0.30,
    padding_ratio: float = 0.05
):
    print(f"=======================================================")
    print(f"🌿 YOLO LEAF LOCALIZATION & CROPPING PIPELINE")
    print(f"=======================================================")
    print(f"📂 Thư mục nguồn      : {src_dir}")
    print(f"📂 Thư mục lá đã cắt  : {out_dir}")
    print(f"🤖 Mô hình YOLO       : {yolo_model_path.name}")
    print(f"🎯 Ngưỡng Confidence : {conf}")

    assert src_dir.exists(), f"Không tìm thấy thư mục nguồn: {src_dir}"
    assert yolo_model_path.exists(), f"Không tìm thấy model YOLO: {yolo_model_path}"

    model = YOLO(str(yolo_model_path))

    class_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    total_raw_images = 0
    total_cropped_leaves = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    for cls_dir in class_dirs:
        dst_cls_dir = out_dir / cls_dir.name
        dst_cls_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted([f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        total_raw_images += len(img_files)

        for img_path in img_files:
            pil_img = Image.open(img_path).convert("RGB")
            img_w, img_h = pil_img.size

            results = model(img_path, conf=conf, verbose=False)[0]
            boxes = results.boxes

            if len(boxes) > 0:
                # Tìm hộp phiến lá lớn nhất / rõ nét nhất
                areas = [
                    float(b.xyxy[0][2] - b.xyxy[0][0]) * float(b.xyxy[0][3] - b.xyxy[0][1])
                    for b in boxes
                ]
                best_idx = int(np.argmax(areas))
                box = boxes[best_idx].xyxy[0].cpu().tolist()

                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1

                # Mở rộng nhẹ padding để không bị mất viền lá
                pad_x = bw * padding_ratio
                pad_y = bh * padding_ratio
                x1_pad = max(0, int(x1 - pad_x))
                y1_pad = max(0, int(y1 - pad_y))
                x2_pad = min(img_w, int(x2 + pad_x))
                y2_pad = min(img_h, int(y2 + pad_y))

                crop_leaf = pil_img.crop((x1_pad, y1_pad, x2_pad, y2_pad))
            else:
                # Nếu ảnh quá mờ / không bắt được box, crop 85% vùng trung tâm
                crop_x1 = int(img_w * 0.075)
                crop_y1 = int(img_h * 0.075)
                crop_x2 = int(img_w * 0.925)
                crop_y2 = int(img_h * 0.925)
                crop_leaf = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            out_file = dst_cls_dir / img_path.name
            crop_leaf.save(out_file, quality=95)
            total_cropped_leaves += 1

    print(f"\n=======================================================")
    print(f"✅ HOÀN TẤT TÁCH LÁ BẰNG YOLO!")
    print(f"📊 Tổng số ảnh gốc xử lý     : {total_raw_images} ảnh")
    print(f"🍃 Tổng số ảnh lá đơn trích xuất: {total_cropped_leaves} ảnh")
    print(f"📁 Lưu tại                  : {out_dir}")
    print(f"=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tách phiến lá bằng YOLOv8")
    parser.add_argument("--src-dir", type=Path, default=ROOT_DIR / "data_plantdoc" / "test")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "data_plantdoc_cropped" / "test")
    parser.add_argument("--yolo-model", type=Path, default=ROOT_DIR / "models" / "yolo" / "yolov8n_leaf.pt")
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    crop_plantdoc_test(args.src_dir, args.out_dir, args.yolo_model, args.conf)
