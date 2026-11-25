#!/usr/bin/env python3
"""
Chuẩn bị bộ dữ liệu thực tế ngoài đồng ruộng PlantDoc (pratikkayal/PlantDoc-Dataset)
và ánh xạ 1-1 toàn bộ 28 lớp về chuẩn 38 lớp PlantVillage (`models/class_indices.json`).

Cách chạy:
    python scripts/prepare_plantdoc.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PLANTDOC_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"

# Ánh xạ chính xác tên thư mục PlantDoc -> tên lớp chuẩn PlantVillage (models/class_indices.json)
PLANTDOC_TO_PLANTVILLAGE: dict[str, str] = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def prepare_plantdoc(raw_dir: Path, out_dir: Path, mapping_path: Path) -> dict:
    if not (raw_dir / "train").exists():
        print(f"🌐 Đang clone PlantDoc-Dataset vào {raw_dir}...")
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", PLANTDOC_REPO_URL, str(raw_dir)], check=True)

    with open(mapping_path, encoding="utf-8") as f:
        valid_classes = set(json.load(f)["class_to_idx"].keys())

    for pv_cls in PLANTDOC_TO_PLANTVILLAGE.values():
        assert pv_cls in valid_classes, f"Class ánh xạ '{pv_cls}' không có trong {mapping_path}"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    stats = {"train": 0, "test": 0, "classes": {}}
    for split in ("train", "test"):
        src_split = raw_dir / split
        if not src_split.exists():
            continue
        for src_cls_dir in sorted(p for p in src_split.iterdir() if p.is_dir()):
            pv_cls = PLANTDOC_TO_PLANTVILLAGE.get(src_cls_dir.name)
            if not pv_cls:
                continue
            dst_cls_dir = out_dir / split / pv_cls
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for img_file in src_cls_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    # Tránh trùng tên file khi copy
                    dst_file = dst_cls_dir / f"pdoc_{img_file.name}"
                    shutil.copy2(img_file, dst_file)
                    count += 1
            stats[split] += count
            stats["classes"].setdefault(pv_cls, {"train": 0, "test": 0})[split] += count

    print(f"✅ Đã chuẩn hóa PlantDoc -> {out_dir}")
    print(f"   • Số lớp ánh xạ 1-1: {len(stats['classes'])} lớp")
    print(f"   • Tập Train (in-the-wild): {stats['train']} ảnh")
    print(f"   • Tập Test  (in-the-wild): {stats['test']} ảnh")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị PlantDoc ánh xạ về 38 lớp PlantVillage")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/plantdoc_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/plantdoc"))
    parser.add_argument("--mapping", type=Path, default=Path("models/class_indices.json"))
    args = parser.parse_args()
    prepare_plantdoc(args.raw_dir, args.out_dir, args.mapping)
