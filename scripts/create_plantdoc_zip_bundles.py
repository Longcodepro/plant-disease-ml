#!/usr/bin/env python3
"""
Đóng gói 2 file ZIP chuẩn cho Google Drive / Google Colab:
1. data_plantdoc_80_20.zip:
   - data_plantdoc/train/ (1,874 ảnh gốc)
   - data_plantdoc/valid/ (468 ảnh gốc)
   - data_plantdoc/test/  (236 ảnh gốc)
2. data_plantdoc_leaves_80_20.zip:
   - data_plantdoc_leaves/train/ (10,959 phiến lá cắt YOLO)
   - data_plantdoc_leaves/valid/ (2,310 phiến lá cắt YOLO)
   - data_plantdoc_leaves/test/  (691 phiến lá cắt YOLO)

Đặc điểm quan trọng:
- Tự động dereference các relative symlink để file zip chứa file ảnh thật 100%,
  không bị lỗi gãy link khi tải về giải nén trên Google Drive / Colab / Windows.
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def zip_dataset(
    archive_name: str,
    base_folder_name: str,
    sources: dict[str, Path]
) -> Path:
    out_zip = ROOT_DIR / archive_name
    print(f"\n📦 Bắt đầu đóng gói: {out_zip.name}...")
    start_time = time.time()

    total_files = 0
    for split_name, src_dir in sources.items():
        assert src_dir.exists(), f"Không tìm thấy thư mục: {src_dir}"
        files = list(src_dir.glob("*/*.*"))
        print(f"   - {split_name:<6}: {len(files):5d} ảnh từ {src_dir}")
        total_files += len(files)

    print(f"   -> Tổng cộng: {total_files} ảnh. Đang nén (ZIP_DEFLATED)...")

    # Tạo file zip
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        added_count = 0
        for split_name, src_dir in sources.items():
            for f in sorted(src_dir.glob("*/*.*")):
                if not f.is_file():
                    continue
                # Giải quyết symlink để lấy file thật
                real_file = f.resolve()
                cls_name = f.parent.name
                file_name = f.name
                
                # Cấu trúc bên trong zip: <base_folder_name>/<split_name>/<cls_name>/<file_name>
                arcname = f"{base_folder_name}/{split_name}/{cls_name}/{file_name}"
                zf.write(real_file, arcname=arcname)
                added_count += 1
                if added_count % 2500 == 0:
                    elapsed = time.time() - start_time
                    print(f"      [Đã nén {added_count}/{total_files} ảnh ({added_count/total_files*100:.1f}%) - {elapsed:.1f}s]")

    elapsed = time.time() - start_time
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Hoàn tất {out_zip.name}!")
    print(f"   - Kích thước : {size_mb:.1f} MB")
    print(f"   - Tổng file  : {added_count} ảnh")
    print(f"   - Thời gian  : {elapsed:.1f} giây")
    return out_zip


def main():
    print("=" * 65)
    print("🚀 TẠO 2 BỘ FILE ZIP PLANTDOC CHUẨN GOOGLE DRIVE / COLAB")
    print("=" * 65)

    # 1. Đóng gói bộ ảnh gốc nguyên tấm
    zip1 = zip_dataset(
        archive_name="data_plantdoc_80_20.zip",
        base_folder_name="data_plantdoc",
        sources={
            "train": ROOT_DIR / "data_plantdoc" / "train_split",
            "valid": ROOT_DIR / "data_plantdoc" / "val_split",
            "test":  ROOT_DIR / "data_plantdoc" / "test",
        }
    )

    # 2. Đóng gói bộ phiến lá cắt YOLO
    zip2 = zip_dataset(
        archive_name="data_plantdoc_leaves_80_20.zip",
        base_folder_name="data_plantdoc_leaves",
        sources={
            "train": ROOT_DIR / "data_plantdoc_leaves" / "train_split",
            "valid": ROOT_DIR / "data_plantdoc_leaves" / "val_split",
            "test":  ROOT_DIR / "data_plantdoc_leaves" / "test",
        }
    )

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ ĐÃ SẴN SÀNG ĐỂ TẢI LÊN GOOGLE DRIVE!")
    print("=" * 65)
    print(f"1. {zip1.name:<32} ({zip1.stat().st_size / (1024*1024):.1f} MB)")
    print(f"2. {zip2.name:<32} ({zip2.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
