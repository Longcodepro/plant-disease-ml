#!/usr/bin/env python3
"""
Kiểm định & Khắc phục Near-Duplicate Data Leakage theo UUID nguồn trên PlantVillage (38 lớp).

Bối cảnh:
- Bộ `data.tar.gz` (New Plant Diseases Dataset - Augmented) đã sinh các biến thể xoay/lật
  (`_180deg`, `_flipTB`, `_new30degFlipLR`, `_newPixel25`...) TRƯỚC khi chia train/valid.
- SHA-256 thông thường chỉ bắt được file giống hệt từng byte, không bắt được biến thể xoay/lật.
- Script này bóc tách UUID ảnh gốc từ tên file (`<uuid>___...`) trên toàn bộ 38 lớp để:
  1. Đo chính xác tỷ lệ rò rỉ theo nhóm (Group/UUID Leakage) giữa `train` và `valid`.
  2. Xuất báo cáo JSON (`notebooks/eda_reports/uuid_leakage_audit.json`).
  3. (Tùy chọn `--build-clean-split`) Tạo bản chia **Group-Stratified 80/20** mới tại `data/clean_split/`
     bảo đảm 0% trùng UUID giữa `train` và `valid` (dùng symlink để không tốn thêm dung lượng).

Cách chạy:
    python scripts/audit_uuid_leakage.py --data-dir data
    python scripts/audit_uuid_leakage.py --data-dir data --build-clean-split --out-dir data/clean_split
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_group_id(filename: str) -> str:
    """Lấy mã ảnh gốc (UUID hoặc tiền tố trước '___' và trước hậu tố augment)."""
    stem = Path(filename).stem
    if "___" in stem:
        return stem.split("___", 1)[0]
    return stem


def audit_and_optionally_split(
    data_dir: Path,
    out_json: Path,
    build_clean_split: bool = False,
    clean_dir: Path | None = None,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy {train_dir} hoặc {valid_dir}")

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    total_train_imgs = 0
    total_valid_imgs = 0
    total_train_uuids = 0
    total_valid_uuids = 0
    total_overlap_uuids = 0
    total_leaked_valid_imgs = 0
    per_class = []

    rng = random.Random(seed)

    for cls in classes:
        tr_groups: dict[str, list[Path]] = defaultdict(list)
        va_groups: dict[str, list[Path]] = defaultdict(list)

        for p in (train_dir / cls).iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                tr_groups[extract_group_id(p.name)].append(p)
                total_train_imgs += 1

        for p in (valid_dir / cls).iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                va_groups[extract_group_id(p.name)].append(p)
                total_valid_imgs += 1

        overlap_ids = set(tr_groups.keys()) & set(va_groups.keys())
        leaked_valid_imgs = sum(len(va_groups[uid]) for uid in overlap_ids)
        va_img_count = sum(len(v) for v in va_groups.values())

        total_train_uuids += len(tr_groups)
        total_valid_uuids += len(va_groups)
        total_overlap_uuids += len(overlap_ids)
        total_leaked_valid_imgs += leaked_valid_imgs

        per_class.append({
            "class_name": cls,
            "train_images": sum(len(v) for v in tr_groups.values()),
            "valid_images": va_img_count,
            "train_unique_uuids": len(tr_groups),
            "valid_unique_uuids": len(va_groups),
            "overlap_uuids": len(overlap_ids),
            "leaked_valid_images": leaked_valid_imgs,
            "leaked_valid_ratio": round(leaked_valid_imgs / va_img_count, 4) if va_img_count else 0.0,
        })

        # Tùy chọn: tạo lại bản chia sạch hoàn toàn không trùng UUID giữa train & valid
        if build_clean_split and clean_dir is not None:
            all_groups: dict[str, list[Path]] = defaultdict(list)
            for uid, paths in tr_groups.items():
                all_groups[uid].extend(paths)
            for uid, paths in va_groups.items():
                all_groups[uid].extend(paths)

            uids = sorted(all_groups.keys())
            rng.shuffle(uids)
            n_val = max(1, int(round(len(uids) * valid_ratio)))
            val_uids = set(uids[:n_val])

            for split_name, selected_uids in (
                ("valid", val_uids),
                ("train", set(uids) - val_uids),
            ):
                dst_cls = clean_dir / split_name / cls
                dst_cls.mkdir(parents=True, exist_ok=True)
                for uid in selected_uids:
                    for src_path in all_groups[uid]:
                        dst_path = dst_cls / src_path.name
                        if not dst_path.exists():
                            os.symlink(src_path.resolve(), dst_path)

    leakage_pct = (total_leaked_valid_imgs / total_valid_imgs * 100.0) if total_valid_imgs else 0.0
    uuid_overlap_pct = (total_overlap_uuids / total_valid_uuids * 100.0) if total_valid_uuids else 0.0

    report = {
        "num_classes": len(classes),
        "total_train_images": total_train_imgs,
        "total_valid_images": total_valid_imgs,
        "total_train_unique_uuids": total_train_uuids,
        "total_valid_unique_uuids": total_valid_uuids,
        "total_overlap_uuids": total_overlap_uuids,
        "valid_uuid_overlap_pct": round(uuid_overlap_pct, 2),
        "leaked_valid_images": total_leaked_valid_imgs,
        "leaked_valid_images_pct": round(leakage_pct, 2),
        "clean_split_created": str(clean_dir) if build_clean_split and clean_dir else None,
        "per_class": sorted(per_class, key=lambda x: x["leaked_valid_ratio"], reverse=True),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print(" 🔎 KIỂM ĐỊNH NEAR-DUPLICATE LEAKAGE THEO UUID (TOÀN BỘ 38 LỚP)")
    print("==================================================================")
    print(f" 📁 Tổng ảnh Train / Valid : {total_train_imgs:,} / {total_valid_imgs:,}")
    print(f" 🆔 UUID gốc Train / Valid : {total_train_uuids:,} / {total_valid_uuids:,}")
    print(f" 🚨 Số UUID trùng Train↔Val: {total_overlap_uuids:,} ({uuid_overlap_pct:.2f}% số UUID tập Valid)")
    print(f" ⚠️  Số ảnh Valid bị rò rỉ : {total_leaked_valid_imgs:,} / {total_valid_imgs:,} ({leakage_pct:.2f}%)")
    if build_clean_split and clean_dir:
        print(f" ✅ Đã tạo Group-Split sạch 0% leakage tại: {clean_dir}")
    print(f" 💾 Báo cáo chi tiết tại   : {out_json}")
    print("==================================================================\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm định & tách lại tập sạch UUID Leakage cho PlantVillage")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-json", type=Path, default=Path("notebooks/eda_reports/uuid_leakage_audit.json"))
    parser.add_argument("--build-clean-split", action="store_true", help="Tạo bản chia Group-Split 0% leakage")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean_split"))
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audit_and_optionally_split(
        data_dir=args.data_dir,
        out_json=args.out_json,
        build_clean_split=args.build_clean_split,
        clean_dir=args.clean_dir,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
