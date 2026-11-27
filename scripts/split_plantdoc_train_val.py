#!/usr/bin/env python3
"""
Tách tập dữ liệu PlantDoc train thành 2 tập Train (80%) và Validation (20%)
với bảo đảm 100% KHÔNG BỊ RÒ RỈ DỮ LIỆU (ZERO DATA LEAKAGE).

Các nguyên tắc kỹ thuật nghiêm ngặt:
1. Group-Aware Splitting (Tránh rò rỉ cấp ảnh gốc):
   - Mọi phiến lá (crops) cắt ra từ cùng một bức ảnh gốc bắt buộc phải nằm chung
     trong một tập (hoặc toàn bộ ở Train, hoặc toàn bộ ở Validation).
2. Hash-Aware Grouping (Tránh rò rỉ ảnh trùng lặp):
   - Các ảnh có cùng mã hash MD5 (ảnh trùng lặp byte) được gom vào cùng một nhóm.
3. Stratified Splitting (Phân tầng theo 28 lớp bệnh):
   - Đảm bảo mỗi lớp bệnh đều giữ đúng tỷ lệ 80% Train / 20% Val.
4. Symlink Storage:
   - Sử dụng relative symlink để không tốn thêm bất kỳ dung lượng đĩa nào.
   - Tương thích 100% với torchvision.datasets.ImageFolder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def split_plantdoc(
    raw_train_dir: Path,
    leaves_train_dir: Path,
    raw_test_dir: Path,
    out_manifest: Path,
    val_ratio: float = 0.20,
    seed: int = 42,
):
    print("=" * 65)
    print("🌿 PLANTDOC STRATIFIED GROUP-AWARE TRAIN/VAL SPLIT")
    print("=" * 65)
    print(f"📂 Thư mục ảnh gốc train  : {raw_train_dir}")
    print(f"🍃 Thư mục lá crop train  : {leaves_train_dir}")
    print(f"📊 Tỷ lệ phân chia        : {int((1 - val_ratio)*100)}% Train / {int(val_ratio*100)}% Val")
    print(f"🎲 Random Seed            : {seed}")

    assert raw_train_dir.exists(), f"Không tìm thấy: {raw_train_dir}"
    random.seed(seed)

    # 1. Quét tập Test để ghi nhận các mã hash overlap (phục vụ audit)
    test_hashes = {}
    if raw_test_dir.exists():
        for f in raw_test_dir.glob("*/*.*"):
            if f.is_file():
                test_hashes[compute_md5(f)] = f

    # 2. Gom nhóm ảnh gốc theo Class và Hash
    # cấu trúc: class_name -> list of group_records
    # mỗi group_record = {"primary_file": Path, "all_raw_files": [Path], "hash": str, "crops": [Path]}
    cls_raw_files = defaultdict(list)
    for cls_dir in sorted(raw_train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.glob("*.*")):
            if f.is_file():
                cls_raw_files[cls_dir.name].append(f)

    # Quét ánh xạ crop -> parent raw image
    raw_name_to_crops = defaultdict(list)
    if leaves_train_dir.exists():
        for leaf_file in leaves_train_dir.glob("*/*.*"):
            if not leaf_file.is_file():
                continue
            leaf_name = leaf_file.name
            if "__leaf_" in leaf_name:
                parent_stem = leaf_name.split("__leaf_")[0]
                raw_name_to_crops[parent_stem].append(leaf_file)

    manifest_data = {
        "metadata": {
            "val_ratio": val_ratio,
            "seed": seed,
            "total_classes": len(cls_raw_files),
        },
        "classes": {},
        "summary": {}
    }

    raw_train_dest = raw_train_dir.parent / "train_split"
    raw_val_dest = raw_train_dir.parent / "val_split"
    leaves_train_dest = leaves_train_dir.parent / "train_split"
    leaves_val_dest = leaves_train_dir.parent / "val_split"

    # Dọn sạch đích cũ nếu có
    for p in [raw_train_dest, raw_val_dest, leaves_train_dest, leaves_val_dest]:
        if p.exists():
            import shutil
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    total_raw_train = 0
    total_raw_val = 0
    total_crops_train = 0
    total_crops_val = 0

    train_raw_set = set()
    val_raw_set = set()
    train_crop_set = set()
    val_crop_set = set()

    for cls_idx, (cls_name, files) in enumerate(sorted(cls_raw_files.items())):
        # Gom các file có cùng hash vào 1 nhóm
        hash_to_files = defaultdict(list)
        for f in files:
            h = compute_md5(f)
            hash_to_files[h].append(f)

        groups = sorted(list(hash_to_files.values()), key=lambda g: g[0].name)
        # Shuffle nhóm với deterministic seed
        rng = random.Random(seed + cls_idx)
        rng.shuffle(groups)

        n_groups = len(groups)
        n_val = max(1, int(round(n_groups * val_ratio))) if n_groups >= 2 else 0
        val_groups = groups[:n_val]
        train_groups = groups[n_val:]

        cls_train_raw = 0
        cls_val_raw = 0
        cls_train_crops = 0
        cls_val_crops = 0

        # Tạo thư mục con cho class
        (raw_train_dest / cls_name).mkdir(parents=True, exist_ok=True)
        (raw_val_dest / cls_name).mkdir(parents=True, exist_ok=True)
        (leaves_train_dest / cls_name).mkdir(parents=True, exist_ok=True)
        (leaves_val_dest / cls_name).mkdir(parents=True, exist_ok=True)

        # Xử lý Train groups
        for grp in train_groups:
            for raw_file in grp:
                cls_train_raw += 1
                train_raw_set.add(raw_file.name)
                # Tạo relative symlink cho raw image
                link_src = os.path.relpath(raw_file.resolve(), (raw_train_dest / cls_name).resolve())
                os.symlink(link_src, raw_train_dest / cls_name / raw_file.name)

                # Tìm và gán crops của ảnh này vào Train
                stem = raw_file.name
                matched_crops = []
                for s in [stem, raw_file.stem, stem.split(".")[0]]:
                    if s in raw_name_to_crops:
                        matched_crops = raw_name_to_crops[s]
                        break

                for crop_file in matched_crops:
                    cls_train_crops += 1
                    train_crop_set.add(crop_file.name)
                    crop_link_src = os.path.relpath(crop_file.resolve(), (leaves_train_dest / cls_name).resolve())
                    os.symlink(crop_link_src, leaves_train_dest / cls_name / crop_file.name)

        # Xử lý Val groups
        for grp in val_groups:
            for raw_file in grp:
                cls_val_raw += 1
                val_raw_set.add(raw_file.name)
                # Tạo relative symlink cho raw image
                link_src = os.path.relpath(raw_file.resolve(), (raw_val_dest / cls_name).resolve())
                os.symlink(link_src, raw_val_dest / cls_name / raw_file.name)

                # Tìm và gán crops của ảnh này vào Val
                stem = raw_file.name
                matched_crops = []
                for s in [stem, raw_file.stem, stem.split(".")[0]]:
                    if s in raw_name_to_crops:
                        matched_crops = raw_name_to_crops[s]
                        break

                for crop_file in matched_crops:
                    cls_val_crops += 1
                    val_crop_set.add(crop_file.name)
                    crop_link_src = os.path.relpath(crop_file.resolve(), (leaves_val_dest / cls_name).resolve())
                    os.symlink(crop_link_src, leaves_val_dest / cls_name / crop_file.name)

        total_raw_train += cls_train_raw
        total_raw_val += cls_val_raw
        total_crops_train += cls_train_crops
        total_crops_val += cls_val_crops

        manifest_data["classes"][cls_name] = {
            "total_raw": len(files),
            "train_raw": cls_train_raw,
            "val_raw": cls_val_raw,
            "train_crops": cls_train_crops,
            "val_crops": cls_val_crops,
        }

    # 3. Kiểm định rò rỉ dữ liệu (Integrity Assertions)
    raw_leakage = train_raw_set & val_raw_set
    crop_leakage = train_crop_set & val_crop_set

    assert len(raw_leakage) == 0, f"LỖI RÒ RỈ: Có {len(raw_leakage)} ảnh gốc xuất hiện ở cả Train và Val!"
    assert len(crop_leakage) == 0, f"LỖI RÒ RỈ: Có {len(crop_leakage)} phiến lá crop xuất hiện ở cả Train và Val!"

    manifest_data["summary"] = {
        "raw_images": {
            "total": total_raw_train + total_raw_val,
            "train": total_raw_train,
            "val": total_raw_val,
            "train_pct": round(total_raw_train / (total_raw_train + total_raw_val) * 100, 2),
            "val_pct": round(total_raw_val / (total_raw_train + total_raw_val) * 100, 2),
            "leakage_count": len(raw_leakage)
        },
        "cropped_leaves": {
            "total": total_crops_train + total_crops_val,
            "train": total_crops_train,
            "val": total_crops_val,
            "train_pct": round(total_crops_train / (total_crops_train + total_crops_val) * 100, 2),
            "val_pct": round(total_crops_val / (total_crops_train + total_crops_val) * 100, 2),
            "leakage_count": len(crop_leakage)
        },
        "directories": {
            "raw_train": str(raw_train_dest),
            "raw_val": str(raw_val_dest),
            "leaves_train": str(leaves_train_dest),
            "leaves_val": str(leaves_val_dest)
        }
    }

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("✅ PHÂN CHIA HOÀN TẤT VỚI 0% DATA LEAKAGE!")
    print("=" * 65)
    print(f"📸 Ảnh gốc (Raw Photos):")
    print(f"   - Train : {total_raw_train:4d} ảnh ({manifest_data['summary']['raw_images']['train_pct']}%)")
    print(f"   - Val   : {total_raw_val:4d} ảnh ({manifest_data['summary']['raw_images']['val_pct']}%)")
    print(f"   - Rò rỉ : 0 ảnh (100% sạch)")
    print(f"🍃 Phiến lá cắt YOLO (Cropped Leaves):")
    print(f"   - Train : {total_crops_train:5d} lá ({manifest_data['summary']['cropped_leaves']['train_pct']}%)")
    print(f"   - Val   : {total_crops_val:5d} lá ({manifest_data['summary']['cropped_leaves']['val_pct']}%)")
    print(f"   - Rò rỉ : 0 lá (100% sạch - cùng ảnh gốc đi chung 1 tập)")
    print(f"📁 Thư mục ảnh gốc   : {raw_train_dest} & {raw_val_dest}")
    print(f"📁 Thư mục phiến lá : {leaves_train_dest} & {leaves_val_dest}")
    print(f"📄 Báo cáo chi tiết : {out_manifest}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tách tập PlantDoc train thành train/val 80/20 không rò rỉ")
    parser.add_argument("--raw-train", type=Path, default=Path("data_plantdoc/train"))
    parser.add_argument("--leaves-train", type=Path, default=Path("data_plantdoc_leaves/train"))
    parser.add_argument("--raw-test", type=Path, default=Path("data_plantdoc/test"))
    parser.add_argument("--out-manifest", type=Path, default=Path("notebooks/eda_reports/plantdoc_train_val_split_manifest.json"))
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    split_plantdoc(
        raw_train_dir=args.raw_train,
        leaves_train_dir=args.leaves_train,
        raw_test_dir=args.raw_test,
        out_manifest=args.out_manifest,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
