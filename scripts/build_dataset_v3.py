#!/usr/bin/env python3
"""
Dựng bộ data_v3: MÔ PHỎNG bộ "New Plant Diseases Dataset (Augmented)" — đủ 14 kiểu ảnh nhân bản
của bộ cũ — nhưng xây trên bộ GỐC (github spMohanty/PlantVillage-Dataset) và ĐÚNG THỨ TỰ:

    gom lá → chia (data_v2, scripts/split_plantvillage.py) → nhân bản CHỈ train (script này)

Thêm so với bộ cũ:
  * train_grayscale/, train_segmented/ : bản xám / xóa nền của ẢNH TRAIN (cùng chiếc lá → không lọt).
  * PlantDoc (ảnh ngoài đồng, CC BY 4.0): lọc ảnh trùng (byte + dHash ≤ 2), bỏ ảnh mâu thuẫn nhãn,
    chia 70/15/15 theo nhóm → phần train trộn vào train/, còn lại → valid_outdoor/, test_outdoor/.

Công thức nhân bản của bộ cũ (suy ra từ 87.867 tên file + so từng điểm ảnh với ảnh gốc):
  lớp có n ảnh train thật < TARGET → mỗi ảnh nhận k = min(14, ceil(TARGET/n) − 1) bản, lấy k kiểu đầu
  trong OLD_TYPES (thứ tự theo tần suất dùng trong bộ cũ). Lớp ≥ TARGET: không nhân.

Cách dùng:
    python3 scripts/build_dataset_v3.py --dry-run
    python3 scripts/build_dataset_v3.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data_v2"                                   # bản chia sạch (ảnh màu thật)
RAW = ROOT / "data" / "plantvillage_raw" / "raw"           # color / grayscale / segmented
PLANTDOC = ROOT / "data" / "plantdoc_raw"
DST = ROOT / "data_v3"
TARGET = 2000
SPLITS = ("train", "valid", "test")

# 14 kiểu của bộ cũ, theo tần suất dùng (bộ cũ thêm kiểu theo đúng thứ tự này khi lớp càng ít ảnh)
OLD_TYPES = ["_flipLR", "_180deg", "_flipTB", "_new30degFlipLR", "_90deg", "_270deg", "_newPixel25",
             "_new30degFlipTB", "_newGRR", "_new200degFlipLR", "_new200degFlipTB", "_new90degFlipLR",
             "_new90degFlipTB", "_newGGR"]

# PlantDoc → 38 lớp PlantVillage (approx = ghép gần đúng)
PLANTDOC_MAP = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust", "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot", "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy", "grape leaf black rot": "Grape___Black_rot",
}
PLANTDOC_APPROX = {"Bell_pepper leaf spot"}
PLANTDOC_PREFIX = "plantdoc__"


# ----------------------------------------------------------------------------- 14 kiểu biến đổi
def apply_old_type(img: Image.Image, t: str) -> Image.Image:
    """Tái hiện đúng từng kiểu ảnh của bộ cũ (đối chiếu từng điểm ảnh với mẫu của bộ cũ)."""
    if t == "_flipLR":
        return ImageOps.mirror(img)
    if t == "_flipTB":
        return ImageOps.flip(img)
    if t == "_180deg":
        return img.transpose(Image.Transpose.ROTATE_180)
    if t == "_90deg":
        return img.transpose(Image.Transpose.ROTATE_90)       # ngược chiều kim đồng hồ
    if t == "_270deg":
        return img.transpose(Image.Transpose.ROTATE_270)
    m = re.fullmatch(r"_new(\d+)degFlip(LR|TB)", t)
    if m:                                                      # lật trước, xoay THUẬN chiều kim đồng hồ, góc đen
        flipped = ImageOps.mirror(img) if m.group(2) == "LR" else ImageOps.flip(img)
        return flipped.rotate(-int(m.group(1)), resample=Image.Resampling.BICUBIC)
    a = np.asarray(img)
    if t == "_newGRR":                                         # đỏ ← xanh lá, xanh lá ← đỏ, xanh dương ← đỏ
        return Image.fromarray(np.ascontiguousarray(a[..., [1, 0, 0]]))
    if t == "_newGGR":                                         # xanh dương ← xanh lá
        return Image.fromarray(np.ascontiguousarray(a[..., [0, 1, 1]]))
    if t == "_newPixel25":                                     # tăng sáng mạnh: x·2 + 35
        return Image.fromarray(np.clip(a.astype(np.float32) * 2.0 + 35, 0, 255).astype(np.uint8))
    raise ValueError(t)


def plan_types(n_real: int, target: int = TARGET) -> list[str]:
    """k kiểu nhân bản cho mỗi ảnh của một lớp có n_real ảnh train thật."""
    if n_real >= target or n_real == 0:
        return []
    k = min(len(OLD_TYPES), math.ceil(target / n_real) - 1)
    return OLD_TYPES[:k]


# ----------------------------------------------------------------------------- PlantDoc
def _dhash(im: Image.Image, n: int = 8) -> int:
    g = np.asarray(im.convert("L").resize((n + 1, n), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, :-1] > g[:, 1:]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _hash_split(key: str) -> str:
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return "train" if h < 0.70 else "valid" if h < 0.85 else "test"


def plan_plantdoc(root: Path, near: int = 2) -> dict:
    files = []
    for part in ("train", "test"):
        for cdir in sorted((root / part).iterdir()):
            for p in sorted(cdir.iterdir()):
                if p.is_file():
                    files.append((cdir.name, p))
    sha, dh = [], []
    for _, p in files:
        sha.append(hashlib.sha256(p.read_bytes()).hexdigest())
        with Image.open(p) as im:
            dh.append(_dhash(im))
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_sha = defaultdict(list)
    for i, s in enumerate(sha):
        by_sha[s].append(i)
    for idx in by_sha.values():
        for j in idx[1:]:
            union(idx[0], j)
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if bin(dh[i] ^ dh[j]).count("1") <= near:
                union(i, j)

    groups = defaultdict(list)
    for i in range(len(files)):
        groups[find(i)].append(i)

    rows, dropped_conflict, dropped_dup = [], [], 0
    for members in groups.values():
        classes = {files[i][0] for i in members}
        if len(classes) > 1:                                    # cùng ảnh mà 2 nhãn → không biết nhãn nào đúng
            dropped_conflict += [f"{files[i][0]}/{files[i][1].name}" for i in members]
            continue
        key = min(sha[i] for i in members)
        split = _hash_split("plantdoc/" + key)
        seen_sha = set()
        for i in members:
            if sha[i] in seen_sha:                               # bản trùng byte → giữ 1
                dropped_dup += 1
                continue
            seen_sha.add(sha[i])
            pd_class, p = files[i]
            ext = p.suffix.lower() if p.suffix.lower() in (".jpg", ".jpeg", ".png") else ".jpg"
            rows.append({"split": split, "pd_class": pd_class, "class": PLANTDOC_MAP[pd_class],
                         "src": p, "file": f"{PLANTDOC_PREFIX}{sha[i][:16]}{ext}", "group": key[:16],
                         "approx_label": pd_class in PLANTDOC_APPROX, "sha256": sha[i]})
    return {"rows": rows, "n_files": len(files), "dropped_conflict": dropped_conflict,
            "dropped_byte_dup": dropped_dup, "n_groups_kept": len({r["group"] for r in rows})}


def _saved_name(r: dict) -> str:
    """Tên file PlantDoc sau khi chép (ảnh .png được đổi sang .jpg)."""
    return r["file"] if r["src"].suffix.lower() in (".jpg", ".jpeg") else str(Path(r["file"]).with_suffix(".jpg"))


# ----------------------------------------------------------------------------- grayscale / segmented
def _photo_id(name: str) -> str:
    s = re.sub(r"\.(jpe?g|png)$", "", name, flags=re.I).replace("_final_masked", "")
    return s.split("___")[-1].strip().lower()


def variant_index(variant: str) -> dict:
    idx = defaultdict(list)
    for cdir in sorted((RAW / variant).iterdir()):
        for p in sorted(cdir.iterdir()):
            idx[(cdir.name, _photo_id(p.name))].append(p)
    return idx


# ----------------------------------------------------------------------------- ghi ra đĩa
def _aug_job(args):
    src, dst, t = args
    with Image.open(src) as im:
        img = im.convert("RGB")
    apply_old_type(img, t).save(dst, quality=95)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    classes = sorted(p.name for p in (CLEAN / "train").iterdir() if p.is_dir())
    clean_rows = list(csv.DictReader(open(CLEAN / "manifest.csv", encoding="utf-8")))

    print("🌿 PlantDoc: lọc trùng & chia theo nhóm …")
    pd = plan_plantdoc(PLANTDOC)
    pd_split = Counter(r["split"] for r in pd["rows"])
    print(f"   {pd['n_files']} ảnh → giữ {len(pd['rows'])} ({pd['n_groups_kept']} nhóm) | bỏ trùng byte "
          f"{pd['dropped_byte_dup']} | bỏ mâu thuẫn nhãn {len(pd['dropped_conflict'])} | {dict(pd_split)}")

    # ảnh train thật mỗi lớp = PlantVillage train + PlantDoc train
    train_real = defaultdict(list)                       # lớp → [(nguồn, tên file đích)]
    for r in clean_rows:
        if r["split"] == "train":
            train_real[r["class"]].append((CLEAN / "train" / r["class"] / r["file"], r["file"]))
    for r in pd["rows"]:
        if r["split"] == "train":
            train_real[r["class"]].append((r["src"], _saved_name(r)))

    aug_plan, per_class = [], {}
    for c in classes:
        types = plan_types(len(train_real[c]), args.target)
        for src, name in train_real[c]:
            stem, ext = os.path.splitext(name)
            for t in types:
                aug_plan.append((c, src, f"{stem}{t}{ext if ext.lower() != '.png' else '.jpg'}", t, name))
        per_class[c] = {"train_real": len(train_real[c]), "types": len(types),
                        "train_total": len(train_real[c]) * (1 + len(types))}
    type_counts = Counter(t for *_, t, _ in aug_plan)
    print(f"🎨 Nhân bản kiểu bộ cũ: {len(aug_plan):,} ảnh | đủ 14 kiểu: {len(type_counts) == 14}")
    for t in OLD_TYPES:
        print(f"   {type_counts[t]:6d}  {t}")

    print("🩶 Ghép ảnh xám / xóa nền cho ảnh train …")
    variants = {}
    for v in ("grayscale", "segmented"):
        idx = variant_index(v)
        pairs, missing = [], 0
        for r in clean_rows:
            if r["split"] != "train":
                continue
            cands = idx.get((r["class"], _photo_id(r["file"])), [])
            if not cands:
                missing += 1
                continue
            stem = os.path.splitext(r["file"])[0]
            for k, p in enumerate(cands):
                suffix = "" if k == 0 else f"_{k}"
                pairs.append((r["class"], p, f"{stem}{suffix}{p.suffix.lower()}"))
        variants[v] = pairs
        print(f"   {v}: {len(pairs):,} ảnh | ảnh train không có bản {v}: {missing}")

    totals = {
        "train": sum(v["train_total"] for v in per_class.values()),
        "train_grayscale": len(variants["grayscale"]), "train_segmented": len(variants["segmented"]),
        "valid": sum(r["split"] == "valid" for r in clean_rows), "test": sum(r["split"] == "test" for r in clean_rows),
        "valid_outdoor": pd_split["valid"], "test_outdoor": pd_split["test"],
    }
    print("📦 Tổng:", totals)
    if args.dry_run:
        print("(dry-run) Không ghi gì.")
        return 0
    if args.dst.exists():
        print(f"❌ {args.dst} đã tồn tại — xóa hoặc chọn --dst khác.", file=sys.stderr)
        return 1

    dst = args.dst
    for part in ("train", "valid", "test", "train_grayscale", "train_segmented"):
        for c in classes:
            (dst / part / c).mkdir(parents=True, exist_ok=True)
    for split, part in (("valid", "valid_outdoor"), ("test", "test_outdoor")):
        for c in sorted({r["class"] for r in pd["rows"] if r["split"] == split}):   # không tạo thư mục rỗng
            (dst / part / c).mkdir(parents=True, exist_ok=True)

    print("📁 Chép ảnh thật …")
    for r in clean_rows:
        shutil.copy2(CLEAN / r["split"] / r["class"] / r["file"], dst / r["split"] / r["class"] / r["file"])
    pd_dest = {"train": "train", "valid": "valid_outdoor", "test": "test_outdoor"}
    for r in pd["rows"]:
        out = dst / pd_dest[r["split"]] / r["class"] / _saved_name(r)
        if r["src"].suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(r["src"], out)
        else:
            with Image.open(r["src"]) as im:
                im.convert("RGB").save(out, quality=95)
    for v, pairs in variants.items():
        for c, p, name in pairs:
            shutil.copy2(p, dst / f"train_{v}" / c / name)

    print(f"🎨 Sinh {len(aug_plan):,} ảnh nhân bản ({args.workers} tiến trình) …")
    jobs = [(str(src), str(dst / "train" / c / name), t) for c, src, name, t, _ in aug_plan]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(_aug_job, jobs, chunksize=64), 1):
            if i % 20000 == 0:
                print(f"   … {i:,}/{len(jobs):,}")

    # manifest & báo cáo
    shutil.copy2(CLEAN / "manifest.csv", dst / "manifest_plantvillage.csv")
    shutil.copy2(CLEAN / "split_report.json", dst / "split_report_plantvillage.json")
    with open(dst / "manifest_augment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "augmented_file", "source_train_file", "old_type"])
        for c, _, name, t, src_name in aug_plan:
            w.writerow([c, name, src_name, t])
    with open(dst / "manifest_plantdoc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "class", "file", "plantdoc_class", "plantdoc_original_file", "group", "approx_label"])
        for r in pd["rows"]:
            w.writerow([pd_dest[r["split"]], r["class"], _saved_name(r), r["pd_class"],
                        f"{r['src'].parent.parent.name}/{r['pd_class']}/{r['src'].name}", r["group"], r["approx_label"]])
    report = {
        "order": "group by leaf -> split 70/15/15 -> augment TRAIN only",
        "target_per_class": args.target, "totals": totals, "per_class_train": per_class,
        "old_types_used": {t: type_counts[t] for t in OLD_TYPES},
        "plantdoc": {"source_files": pd["n_files"], "kept": len(pd["rows"]), "groups": pd["n_groups_kept"],
                     "dropped_byte_duplicates": pd["dropped_byte_dup"],
                     "dropped_label_conflicts": pd["dropped_conflict"], "split": dict(pd_split),
                     "approx_mapping": sorted(PLANTDOC_APPROX)},
    }
    (dst / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Xong: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
