"""
Exploratory Data Analysis (EDA) & Data Health Check Script
Dự án: Chẩn đoán bệnh trên lá cây nông nghiệp (PlantVillage Dataset)
Tuân thủ nguyên tắc từ:
- Designing Machine Learning Systems (Chip Huyen) - Chương 3, 4, 5
- Hands-On Machine Learning (Aurélien Géron) - Chương 2, 13, 14
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path("notebooks") / "eda_reports"

def inspect_classes(train_dir: Path, valid_dir: Path) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Kiểm tra tính nhất quán giữa tập train và valid:
    - Danh sách class có khớp nhau 100% không?
    - Đếm số lượng ảnh trên mỗi class.
    """
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục {train_dir} hoặc {valid_dir}")

    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    valid_classes = sorted([d.name for d in valid_dir.iterdir() if d.is_dir()])

    if train_classes != valid_classes:
        diff_train = set(train_classes) - set(valid_classes)
        diff_valid = set(valid_classes) - set(train_classes)
        raise ValueError(
            f"Lỗi rò rỉ hoặc thiếu nhãn! Train và Valid không đồng nhất: "
            f"Chỉ có trong train: {diff_train}, chỉ có trong valid: {diff_valid}"
        )

    logger.info("Xác nhận: Danh sách nhãn giữa Train và Valid hoàn toàn khớp nhau (%d lớp).", len(train_classes))

    train_counts = {c: len(list((train_dir / c).glob("*.*"))) for c in train_classes}
    valid_counts = {c: len(list((valid_dir / c).glob("*.*"))) for c in valid_classes}

    return train_classes, train_counts, valid_counts

def verify_image_health(directory: Path) -> Tuple[int, int, set, set]:
    """
    Kiểm tra tính toàn vẹn của dữ liệu (Data Integrity Check):
    - Phát hiện ảnh bị hỏng (corrupted files).
    - Kiểm tra dải kích thước (resolutions) và không gian màu (color modes).
    """
    total_images = 0
    corrupt_images = 0
    resolutions = set()
    modes = set()

    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                total_images += 1
                img_path = os.path.join(root, f)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    # Mở lại để đọc kích thước và mode (sau verify cần mở lại)
                    with Image.open(img_path) as img:
                        resolutions.add(img.size)
                        modes.add(img.mode)
                except Exception as e:
                    corrupt_images += 1
                    logger.warning("Phát hiện ảnh lỗi: %s | Chi tiết: %s", img_path, e)

    return total_images, corrupt_images, resolutions, modes

def export_metadata_and_visualizations(
    classes: List[str],
    train_counts: Dict[str, int],
    valid_counts: Dict[str, int],
    output_dir: Path
) -> pd.DataFrame:
    """
    Xuất bảng thống kê phân phối nhãn (Class Distribution) và vẽ biểu đồ.
    Lưu file mapping class -> index cho Backend phục vụ Inference (tránh hardcode).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tạo DataFrame thống kê
    data = []
    for c in classes:
        t_cnt = train_counts[c]
        v_cnt = valid_counts[c]
        total = t_cnt + v_cnt
        val_ratio = v_cnt / total if total > 0 else 0
        data.append({
            "class_name": c,
            "train_count": t_cnt,
            "valid_count": v_cnt,
            "total_count": total,
            "valid_ratio": round(val_ratio, 4)
        })

    df = pd.DataFrame(data).sort_values(by="total_count", ascending=False)
    csv_path = output_dir / "class_distribution.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Đã lưu bảng phân phối nhãn tại: %s", csv_path)

    # 2. Lưu mapping class -> id thành JSON phục vụ Backend FastAPI
    class_to_idx = {c: idx for idx, c in enumerate(classes)}
    idx_to_class = {idx: c for idx, c in enumerate(classes)}
    mapping_path = Path("models") / "class_indices.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=4)
    logger.info("Đã lưu mapping nhãn (38 lớp) vào: %s", mapping_path)

    # 3. Phân tích mức độ mất cân bằng (Imbalance Ratio)
    max_count = df["train_count"].max()
    min_count = df["train_count"].min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    logger.info("Mức độ mất cân bằng (Imbalance Ratio = Max/Min): %.2f (%d / %d)", imbalance_ratio, max_count, min_count)

    # 4. Trực quan hóa phân phối nhãn
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df, y="class_name", x="train_count", color="#2b5c8f", label="Train")
    sns.barplot(data=df, y="class_name", x="valid_count", color="#e27c38", label="Validation")
    plt.title("Phân phối số lượng mẫu trên 38 lớp (PlantVillage Dataset)", fontsize=14, fontweight="bold")
    plt.xlabel("Số lượng mẫu (Images)", fontsize=12)
    plt.ylabel("Tên lớp bệnh / cây trồng", fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Đã xuất biểu đồ phân phối nhãn tại: %s", plot_path)

    return df

def main():
    logger.info("=== BẮT ĐẦU QUY TRÌNH THẨM ĐỊNH DỮ LIỆU & EDA ===")
    classes, train_counts, valid_counts = inspect_classes(TRAIN_DIR, VALID_DIR)

    logger.info("Đang kiểm tra tính toàn vẹn của tập Train...")
    t_tot, t_cor, t_res, t_modes = verify_image_health(TRAIN_DIR)
    logger.info("Train: Tổng = %d, Lỗi = %d, Kích thước mẫu = %s, Hệ màu = %s", t_tot, t_cor, t_res, t_modes)

    logger.info("Đang kiểm tra tính toàn vẹn của tập Valid...")
    v_tot, v_cor, v_res, v_modes = verify_image_health(VALID_DIR)
    logger.info("Valid: Tổng = %d, Lỗi = %d, Kích thước mẫu = %s, Hệ màu = %s", v_tot, v_cor, v_res, v_modes)

    df_summary = export_metadata_and_visualizations(classes, train_counts, valid_counts, OUTPUT_DIR)

    print("\n--- BẢNG THỐNG KÊ 5 LỚP CÓ NHIỀU MẪU NHẤT VÀ ÍT MẪU NHẤT ---")
    print("Top 5 nhiều mẫu nhất:")
    print(df_summary[["class_name", "train_count", "valid_count", "total_count"]].head(5).to_string(index=False))
    print("\nTop 5 ít mẫu nhất:")
    print(df_summary[["class_name", "train_count", "valid_count", "total_count"]].tail(5).to_string(index=False))
    logger.info("=== HOÀN TẤT EDA VÀ KIỂM TRA TÍNH TOÀN VẸN DỮ LIỆU ===")

if __name__ == "__main__":
    main()
