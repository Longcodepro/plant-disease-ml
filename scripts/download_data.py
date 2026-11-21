#!/usr/bin/env python3
"""
Script tự động tải và giải nén Dataset cho dự án Plant Disease ML.
- Hỗ trợ định dạng: .zip, .tar.gz, .tgz, .tar
- Nguồn: Google Drive (hoặc file nén có sẵn trên máy)
- Đích: Tự động giải nén và phân phối chuẩn vào thư mục data/ (data/train, data/valid, data/test)
"""

import os
import sys
import shutil
import zipfile
import tarfile
import argparse
from pathlib import Path

# Cấu hình ID hoặc link Google Drive mặc định của dự án
DEFAULT_GDRIVE_FILE_ID = "1HKKtjCbc0BcdcNAMA4ualjxXhOprBaqc"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def extract_archive(archive_path: Path, extract_to: Path):
    """Giải nén đa định dạng (.zip, .tar.gz, .tgz, .tar)."""
    name = archive_path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        print(f"📦 Đang giải nén file tar.gz: {archive_path.name}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_to)
    elif name.endswith(".tar"):
        print(f"📦 Đang giải nén file tar: {archive_path.name}...")
        with tarfile.open(archive_path, "r:") as tar:
            tar.extractall(extract_to)
    elif name.endswith(".zip"):
        print(f"📦 Đang giải nén file zip: {archive_path.name}...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {archive_path.name}")


def extract_and_organize(archive_path: Path, target_data_dir: Path):
    """
    Giải nén an toàn và tự động tổ chức chuẩn vào target_data_dir (data/):
    Hỗ trợ linh hoạt:
    - Nếu file nén chứa trực tiếp train/, valid/, test/ (hoặc data/train, data/valid, v.v.)
    """
    target_data_dir.mkdir(parents=True, exist_ok=True)
    temp_extract_dir = target_data_dir / "_temp_extract"
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    extract_archive(archive_path, temp_extract_dir)

    print("🚚 Đang quét và phân phối các tập dữ liệu vào đúng vị trí data/...")

    found_train = None
    found_valid = None
    found_test = None

    # Quét tìm kiếm thư mục train, valid, test trong cây thư mục vừa giải nén
    for root, dirs, files in os.walk(temp_extract_dir):
        root_path = Path(root)
        name_lower = root_path.name.lower()

        # Nhận diện train
        if name_lower == "train" and found_train is None:
            if any(p.is_dir() for p in root_path.iterdir()):
                found_train = root_path

        # Nhận diện valid / val / validation
        elif name_lower in ("valid", "val", "validation") and found_valid is None:
            if any(p.is_dir() for p in root_path.iterdir()):
                found_valid = root_path

        # Nhận diện test
        elif name_lower == "test" and found_test is None:
            if any(p.is_file() for p in root_path.iterdir()):
                found_test = root_path

    # Di chuyển train vào data/train
    if found_train:
        dest_train = target_data_dir / "train"
        if dest_train.exists():
            shutil.rmtree(dest_train)
        shutil.move(str(found_train), str(dest_train))
        print("  ✅ Đã sắp xếp: data/train/")
    else:
        print("  ⚠️ Cảnh báo: Không tìm thấy thư mục train!")

    # Di chuyển valid vào data/valid
    if found_valid:
        dest_valid = target_data_dir / "valid"
        if dest_valid.exists():
            shutil.rmtree(dest_valid)
        shutil.move(str(found_valid), str(dest_valid))
        print("  ✅ Đã sắp xếp: data/valid/")
    else:
        print("  ⚠️ Cảnh báo: Không tìm thấy thư mục valid!")

    # Di chuyển test vào data/test
    if found_test:
        dest_test = target_data_dir / "test"
        if dest_test.exists():
            shutil.rmtree(dest_test)
        shutil.move(str(found_test), str(dest_test))
        print("  ✅ Đã sắp xếp: data/test/")
    else:
        print("  ℹ️ Thông tin: Không có thư mục test riêng biệt (không bắt buộc).")

    # Dọn dẹp thư mục tạm
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print("🎉 Hoàn tất! Toàn bộ dữ liệu đã nằm chuẩn chỉnh trong thư mục data/")


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
        help="Đường dẫn đến file nén có sẵn trên máy (.zip, .tar.gz)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        default=None,
        help="Đường link Google Drive chia sẻ file nén"
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Giữ lại file nén sau khi giải nén (mặc định sẽ xóa để tiết kiệm dung lượng)"
    )

    args = parser.parse_args()

    # Tìm file nén có sẵn
    local_archive = None
    if args.file and Path(args.file).exists():
        local_archive = Path(args.file)
    else:
        # Tự động tìm các file nén trong thư mục gốc
        valid_extensions = (".tar.gz", ".tgz", ".zip", ".tar")
        for f in ROOT_DIR.iterdir():
            if f.is_file() and any(f.name.lower().endswith(ext) for ext in valid_extensions):
                if f.name not in ("dataset.zip", "dataset.tar.gz"):
                    print(f"🔍 Tự động phát hiện file nén sẵn có: {f.name}")
                    local_archive = f
                    break

    if local_archive:
        extract_and_organize(local_archive, DATA_DIR)
        return

    # Tải từ Google Drive
    source_url = args.url or DEFAULT_GDRIVE_FILE_ID
    if source_url == "YOUR_GDRIVE_FILE_ID_HERE":
        print("❌ Lỗi: Bạn chưa cung cấp link Google Drive!")
        print("👉 Cách dùng:")
        print("   python scripts/download_data.py --url 'LINK_GOOGLE_DRIVE'")
        print("   hoặc đặt file nén (.tar.gz / .zip) vào thư mục gốc rồi chạy: python scripts/download_data.py")
        sys.exit(1)

    dest_archive = ROOT_DIR / "dataset_downloaded"
    download_from_gdrive(source_url, dest_archive)

    if not dest_archive.exists() or dest_archive.stat().st_size == 0:
        print("❌ Lỗi: Quá trình tải file từ Google Drive thất bại.")
        sys.exit(1)

    extract_and_organize(dest_archive, DATA_DIR)

    if not args.keep_archive and dest_archive.exists():
        print("🧹 Dọn dẹp: Đang xóa file nén tạm để giải phóng dung lượng...")
        dest_archive.unlink()

    print("✨ Sẵn sàng huấn luyện mô hình!")


if __name__ == "__main__":
    main()
