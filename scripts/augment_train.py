#!/usr/bin/env python3
"""
Nhân bản (augment) ẢNH TRAIN để cân bằng lớp — làm SAU KHI CHIA, CHỈ TRÊN TRAIN.

Thứ tự đúng:  gom lá → chia (scripts/split_plantvillage.py) → nhân bản CHỈ train (script này).
valid/ và test/ được chép nguyên vẹn: chỉ chứa ảnh thật, không có bản xoay/lật nào.

Cân bằng lớp: lớp nào trong train có ít hơn --target ảnh thì sinh thêm (target − n) ảnh từ chính
ảnh train của lớp đó (lần lượt từng ảnh gốc); lớp đã đủ thì giữ nguyên.

Mỗi ảnh sinh ra dùng một "hạt giống" tính từ (lớp, tên file gốc, số thứ tự) → chạy lại ở máy nào cũng
ra đúng các ảnh đó. Phép biến đổi (Pillow): xoay 0/90/180/270°, lật ngang/dọc, xoay nhẹ ±20° kèm
phóng to để không lộ góc đen, cắt-phóng 80–100%, chỉnh sáng/tương phản/bão hòa ±20%.

Cách dùng:
    python3 scripts/augment_train.py --dry-run
    python3 scripts/augment_train.py                       # data_v2 → data_v2_aug
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT_DIR / "data_v2"
DEFAULT_DST = ROOT_DIR / "data_v2_aug"
AUG_TAG = "_aug"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _rng(class_name: str, filename: str, k: int) -> random.Random:
    seed = hashlib.sha256(f"{class_name}/{filename}/{k}".encode()).digest()[:8]
    return random.Random(int.from_bytes(seed, "big"))


def augment_image(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    """Một phiên bản biến đổi ngẫu nhiên (theo rng) của ảnh. Trả về (ảnh, mô tả phép biến đổi)."""
    w, h = img.size
    ops = []
    rot90 = rng.choice([0, 90, 180, 270])
    if rot90:
        img = img.rotate(rot90, expand=True)
        ops.append(f"rot{rot90}")
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
        ops.append("flipLR")
    if rng.random() < 0.3:
        img = ImageOps.flip(img)
        ops.append("flipTB")
    angle = rng.uniform(-20, 20)
    # xoay nhẹ rồi cắt phần giữa đủ nhỏ để không còn góc đen
    rad = math.radians(abs(angle))
    keep = 1 / (math.cos(rad) + math.sin(rad))
    img = img.rotate(angle, resample=Image.BICUBIC)
    # vùng an toàn (không có góc đen) là hình vuông ở giữa, cạnh = keep × kích thước ảnh
    safe_w, safe_h = int(img.width * keep), int(img.height * keep)
    x0, y0 = (img.width - safe_w) // 2, (img.height - safe_h) // 2
    zoom = rng.uniform(0.8, 1.0)  # cắt 80–100% vùng an toàn rồi phóng lại kích thước cũ
    cw, ch = int(safe_w * zoom), int(safe_h * zoom)
    left = x0 + rng.randint(0, safe_w - cw)
    top = y0 + rng.randint(0, safe_h - ch)
    img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    ops.append(f"rot{angle:+.1f}_zoom{zoom:.2f}")
    for name, enh in (("b", ImageEnhance.Brightness), ("c", ImageEnhance.Contrast), ("s", ImageEnhance.Color)):
        f = rng.uniform(0.8, 1.2)
        img = enh(img).enhance(f)
        ops.append(f"{name}{f:.2f}")
    return img, " ".join(ops)


def plan(src: Path, target: int) -> dict:
    train = src / "train"
    classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    jobs = []  # (lớp, file gốc, k, tên file mới)
    per_class = {}
    for c in classes:
        originals = sorted(p.name for p in (train / c).iterdir()
                           if p.suffix.lower() in IMAGE_EXTS and AUG_TAG not in p.stem)
        need = max(0, target - len(originals))
        for i in range(need):
            orig = originals[i % len(originals)]
            k = i // len(originals) + 1
            stem, ext = os.path.splitext(orig)
            jobs.append((c, orig, k, f"{stem}{AUG_TAG}{k:02d}{ext}"))
        per_class[c] = {"original": len(originals), "augmented": need, "total": len(originals) + need}
    return {"classes": classes, "jobs": jobs, "per_class": per_class}


def _run_job(args: tuple) -> tuple[str, str, str, str]:
    src_train, dst_train, c, orig, k, new_name = args
    with Image.open(Path(src_train) / c / orig) as im:
        img = im.convert("RGB")
    out, ops = augment_image(img, _rng(c, orig, k))
    out.save(Path(dst_train) / c / new_name, quality=95)
    return c, new_name, orig, ops


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, copy_function=shutil.copy2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--target", type=int, default=2000, help="số ảnh train tối thiểu mỗi lớp")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    p = plan(args.src, args.target)
    n_orig = sum(v["original"] for v in p["per_class"].values())
    print(f"Train gốc: {n_orig} | sinh thêm: {len(p['jobs'])} | train sau nhân bản: {n_orig + len(p['jobs'])}")
    lo = min(p["per_class"].items(), key=lambda kv: kv[1]["original"])
    print(f"Lớp ít nhất: {lo[0]} {lo[1]}")
    if args.dry_run:
        print("(dry-run) Không ghi gì.")
        return 0
    if args.dst.exists():
        print(f"❌ {args.dst} đã tồn tại — xóa hoặc chọn --dst khác.", file=sys.stderr)
        return 1

    print("📁 Chép train/valid/test gốc …")
    for split in ("train", "valid", "test"):
        _copy_tree(args.src / split, args.dst / split)
    for f in ("manifest.csv", "split_report.json", "README.txt"):
        if (args.src / f).exists():
            shutil.copy2(args.src / f, args.dst / f)

    print(f"🎨 Sinh {len(p['jobs'])} ảnh train ({args.workers} tiến trình) …")
    work = [(str(args.src / "train"), str(args.dst / "train"), *j) for j in p["jobs"]]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_run_job, work, chunksize=64), 1):
            rows.append(r)
            if i % 5000 == 0:
                print(f"  … {i}/{len(work)}")

    with open(args.dst / "augment_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "augmented_file", "source_train_file", "ops"])
        w.writerows(sorted(rows))
    report = {"target_per_class": args.target, "train_original": n_orig,
              "train_augmented_added": len(rows), "per_class_train": p["per_class"],
              "note": "Chỉ train được nhân bản; valid/test là ảnh thật, giống hệt data_v2."}
    (args.dst / "augment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Xong: {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
