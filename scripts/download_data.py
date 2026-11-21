#!/usr/bin/env python3
"""
Script tự động tải và giải nén Dataset cho dự án Plant Disease ML.
- Nguồn: Google Drive (hoặc file archive.zip có sẵn trên máy)
- Đích: Tự động phân phối chuẩn vào thư mục data/ (data/train, data/valid, data/test)
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

# Cấu hình ID hoặc link Google Drive mặc định (thay thế ID này khi có link chính thức)
DEFAULT_GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def extract_and_organize(zip_path: Path, target_data_dir: Path):
    """
    Giải nén và tổ chức dữ liệu chuẩn vào target_data_dir:
    - data/train
    - data/valid
    - data/test
    """
    print(f"📦 Đang đọc và giải nén {zip_path.name}...")
    target_data_dir.mkdir(parents=True, exist_ok=True)
    temp_extract_dir = target_data_dir / "_temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        total_files = len(zip_ref.infolist())
        print(f"🔍 Tổng số tệp trong file nén: {total_files}")
        zip_ref.extractall(temp_extract_dir)

    print("🚚 Đang phân loại dữ liệu vào đúng vị trí data/...")

    # Tìm kiếm thư mục train, valid, test trong thư mục tạm vừa bung
    found_train = None
    found_valid = None
    found_test = None

    for root, dirs, files in os.walk(temp_extract_dir):
        root_path = Path(root)
        if root_path.name.lower() == "train" and found_train is None:
            # Kiểm tra xem có chứa các thư mục con phân lớp không
            if any(p.is_dir() for p in root_path.iterdir()):
                found_train = root_path
        elif root_path.name.lower() in ("valid", "val") and found_valid is None:
            if any(p.is_dir() for p in root_path.iterdir()):
                found_valid = root_path
        elif root_path.name.lower() == "test" and found_test is None:
            if any(p.is_file() for p in root_path.iterdir()):
                found_test = root_path

    # Di chuyển train
    if found_train:
        dest_train = target_data_dir / "train"
        if dest_train.exists():
            shutil.rmtree(dest_train)
        shutil.move(str(found_train), str(dest_train))
        print("  ✅ Đã sắp xếp thư mục data/train/")
    else:
        print("  ⚠️ Không tìm thấy thư mục train trong file nén!")

    # Di chuyển valid
    if found_valid:
        dest_valid = target_data_dir / "valid"
        if dest_valid.exists():
            shutil.rmtree(dest_valid)
        shutil.move(str(found_valid), str(dest_valid))
        print("  ✅ Đã sắp xếp thư mục data/valid/")
    else:
        print("  ⚠️ Không tìm thấy thư mục valid trong file nén!")

    # Di chuyển test
    if found_test:
        dest_test = target_data_dir / "test"
        if dest_test.exists():
            shutil.rmtree(dest_test)
        shutil.move(str(found_test), str(dest_test))
        print("  ✅ Đã sắp xếp thư mục data/test/")
    else:
        print("  ℹ️ Không có thư mục test riêng biệt (không bắt buộc).")

    # Dọn dẹp thư mục tạm
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print("🎉 Hoàn tất sắp xếp dữ liệu vào thư mục data/!")


def download_from_gdrive(file_id_or_url: str, output_path: Path):
    """Tải file từ Google Drive bằng gdown."""
    try:
        import gdown
    except ImportError:
        print("⚠️ Chưa có thư viện 'gdown'. Đang tự động cài đặt...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    print(f"🌐 Đang kết nối Google Drive để tải file về {output_path}...")
    if "drive.google.com" in file_id_or_url:
        gdown.download(url=file_id_or_url, output=str(output_path), quiet=False, fuzzy=True)
    else:
        gdown.download(id=file_id_or_url, output=str(output_path), quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Tải và tự động giải nén dataset lá cây vào data/")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Đường dẫn đến file zip có sẵn trên máy (nếu đã tải về trước)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default=None,
        help="Đường link Google Drive chia sẻ file zip"
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Giữ lại file zip sau khi giải nén (mặc định sẽ xóa để tiết kiệm dung lượng ổ cứng)"
    )

    args = parser.parse_args()

    # Trường hợp 1: Có file zip sẵn trên máy (chỉ định hoặc tự tìm thấy trong root)
    local_zip = None
    if args.file and Path(args.file).exists():
        local_zip = Path(args.file)
    elif (ROOT_DIR / "archive.zip").exists():
        print("🔍 Tìm thấy file archive.zip sẵn trong thư mục dự án.")
        local_zip = ROOT_DIR / "archive.zip"

    if local_zip:
        extract_and_organize(local_zip, DATA_DIR)
        return

    # Trường hợp 2: Tải từ Google Drive
    source_url = args.url or DEFAULT_GDRIVE_FILE_ID
    if source_url == "YOUR_GDRIVE_FILE_ID_HERE":
        print("❌ Lỗi: Bạn chưa cung cấp link Google Drive!")
        print("👉 Cách dùng:")
        print("   python scripts/download_data.py --url 'LINK_GOOGLE_DRIVE'")
        print("   hoặc đặt file archive.zip vào thư mục gốc rồi chạy: python scripts/download_data.py")
        sys.exit(1)

    dest_zip = ROOT_DIR / "dataset.zip"
    download_from_gdrive(source_url, dest_zip)

    if not dest_zip.exists() or dest_zip.stat().st_size == 0:
        print("❌ Lỗi: Quá trình tải file từ Google Drive thất bại.")
        sys.exit(1)

    extract_and_organize(dest_zip, DATA_DIR)

    if not args.keep_zip and dest_zip.exists():
        print("🧹 Dọn dẹp: Đang xóa file zip tạm để giải phóng dung lượng...")
        dest_zip.unlink()

    print("✨ Sẵn sàng huấn luyện mô hình!")


if __name__ == "__main__":
    main()
