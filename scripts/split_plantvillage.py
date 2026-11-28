#!/usr/bin/env python3
"""
Chia bộ PlantVillage GỐC (chưa nhân bản) thành train / valid / test — ĐÚNG THỨ TỰ, chống rò rỉ.

Thứ tự đúng:  gom ảnh cùng một chiếc lá  →  chia  →  (xoay/lật CHỈ trên train, làm online
trong notebooks/data_pipeline.py — script này KHÔNG nhân bản ảnh nào).

Vì sao phải "gom" trước khi chia?
  - Một chiếc lá được chụp NHIỀU TẤM với mã khác nhau (tới 33 tấm/lá). Tác giả PlantVillage công bố
    bản đồ chính thức `leaf-map.json` (mã chụp → mã lá), vd. "RS_Early.B 7554" và "RS_Early.B 7557"
    đều là lá số 115. ~54.300 ảnh chỉ đến từ ~20.000 chiếc lá.
  - Cùng một chiếc lá được chụp nhiều ngày:  "GHLB Leaf 1.1 Day 6"  và  "GHLB Leaf 1.1 Day 12".
  - Cùng mã chụp xuất hiện 2 lần:            "RS_HL 6274" (2 file, 2 UUID khác nhau).
  - Ảnh trùng hệt nhau từng byte.
  Các ảnh thuộc cùng một nhóm luôn được đưa vào CÙNG một phần.

Chia bằng cách băm (Géron, Hands-On ML, Chương 2): SHA-256 của khóa nhóm → số trong [0, 1)
→ < 0.70 train, < 0.85 valid, còn lại test. Không phụ thuộc máy, thứ tự file hay random seed.

Chỉ dùng thư viện chuẩn (chạy được cả trên Colab).

Cách dùng:
    python3 scripts/split_plantvillage.py --dry-run          # chỉ xem thống kê, không ghi gì
    python3 scripts/split_plantvillage.py                    # ghi ra data_v2/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT_DIR / "data" / "plantvillage_raw" / "raw" / "color"
DEFAULT_DST = ROOT_DIR / "data_v2"
CLASS_INDICES = ROOT_DIR / "models" / "class_indices.json"
DEFAULT_LEAF_MAP = ROOT_DIR / "data" / "plantvillage_raw" / "leaf-map.json"

SPLITS = ("train", "valid", "test")
TRAIN_FRAC, VALID_FRAC = 0.70, 0.15
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Mã bản sao đứng trước "___" (thường là UUID). Phần sau "___" mới là mã chụp của chiếc lá.
_UUID_PREFIX = re.compile(r"^[^_]+___")
_DAY_SUFFIX = re.compile(r"\s+day\s+\d+$", re.I)
_EXT_SUFFIX = re.compile(r"\.(jpe?g|png)$", re.I)


# ----------------------------------------------------------------------------- khóa nhóm
def leaf_key(class_name: str, filename: str) -> str:
    """Khóa "cùng một chiếc lá": <lớp>/<mã chụp> sau khi bỏ UUID, đuôi file, " Day N", chữ hoa.

    >>> leaf_key("Tomato___Late_blight", "d0badc95-d04c-457b-8db8-aa6e198f4bd6___GHLB Leaf 10 Day 6.jpeg")
    'Tomato___Late_blight/ghlb leaf 10'
    """
    stem = _EXT_SUFFIX.sub("", filename)
    stem = _UUID_PREFIX.sub("", stem)
    stem = _DAY_SUFFIX.sub("", stem.strip())
    stem = re.sub(r"\s+", " ", stem).strip().lower()
    return f"{class_name}/{stem}"


def official_leaf_id(class_name: str, filename: str, leaf_map: dict) -> str | None:
    """Mã lá chính thức từ leaf-map.json — cùng logic với loader gốc (plant_village.py của PlantVillage).

    >>> official_leaf_id("Potato___Early_blight", "u___RS_Early.B 7554.JPG",
    ...                  {"rs_early.b 7554": ["Potato___Early_blight:::115.0"]})
    'Potato___Early_blight:::115.0'
    """
    ident = filename.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    ident = _EXT_SUFFIX.sub("", ident.strip()).strip().lower()
    suggestions = leaf_map.get(ident)
    if not suggestions:
        return None
    if len(suggestions) == 1:
        return suggestions[0]
    for sug in suggestions:
        if class_name in sug:
            return sug
    return None


def split_of(group_key: str) -> str:
    """Băm khóa nhóm → train/valid/test. Cùng khóa ⇒ luôn cùng phần, ở mọi máy."""
    h = int.from_bytes(hashlib.sha256(group_key.encode("utf-8")).digest()[:8], "big") / 2**64
    if h < TRAIN_FRAC:
        return "train"
    if h < TRAIN_FRAC + VALID_FRAC:
        return "valid"
    return "test"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # gốc = chuỗi nhỏ hơn → kết quả không phụ thuộc thứ tự duyệt file
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- lập kế hoạch chia
def plan_split(src: Path, leaf_map: dict | None = None) -> dict:
    """Trả về kế hoạch chia (chưa ghi gì ra đĩa)."""
    leaf_map = leaf_map or {}
    classes = sorted(p.name for p in src.iterdir() if p.is_dir())
    files: list[tuple[str, str, Path]] = []  # (lớp, tên file, đường dẫn)
    for c in classes:
        for p in sorted((src / c).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append((c, p.name, p))

    uf = _UnionFind()
    by_hash: dict[str, list[int]] = defaultdict(list)
    n_mapped = 0
    for i, (c, name, p) in enumerate(files):
        k = leaf_key(c, name)
        uf.find(k)
        lid = official_leaf_id(c, name, leaf_map)
        if lid is not None:
            uf.union(k, "leaf-map:" + lid)   # mọi tấm của cùng một lá chính thức → một nhóm
            n_mapped += 1
        by_hash[_sha256_file(p)].append(i)

    # Ảnh trùng byte: giữ 1 tấm, và gom khóa của chúng lại với nhau
    dropped: list[dict] = []
    keep = [True] * len(files)
    for idxs in by_hash.values():
        if len(idxs) > 1:
            first = idxs[0]
            for j in idxs[1:]:
                uf.union(leaf_key(*files[first][:2]), leaf_key(*files[j][:2]))
                keep[j] = False
                dropped.append({"file": f"{files[j][0]}/{files[j][1]}",
                                "duplicate_of": f"{files[first][0]}/{files[first][1]}"})

    rows = []
    for i, (c, name, p) in enumerate(files):
        if not keep[i]:
            continue
        group = uf.find(leaf_key(c, name))
        rows.append({"split": split_of(group), "class": c, "file": name, "group": group, "path": p,
                     "leaf_id": official_leaf_id(c, name, leaf_map)})

    return {"classes": classes, "rows": rows, "dropped_byte_duplicates": dropped,
            "n_source_files": len(files), "n_in_leaf_map": n_mapped}


def verify(plan: dict, expected_classes: list[str] | None) -> list[str]:
    """Các điều KHÔNG được phép xảy ra. Trả về danh sách lỗi (rỗng = ổn)."""
    errors = []
    group_splits: dict[str, set] = defaultdict(set)
    for r in plan["rows"]:
        group_splits[r["group"]].add(r["split"])
    leaked = [g for g, s in group_splits.items() if len(s) > 1]
    if leaked:
        errors.append(f"{len(leaked)} nhóm nằm ở >1 phần, ví dụ {leaked[:3]}")

    # Kiểm tra chéo độc lập (không dùng union-find): khóa lá thô không được nằm ở 2 phần
    raw_splits: dict[str, set] = defaultdict(set)
    for r in plan["rows"]:
        raw_splits[leaf_key(r["class"], r["file"])].add(r["split"])
    raw_leaked = [k for k, s in raw_splits.items() if len(s) > 1]
    if raw_leaked:
        errors.append(f"{len(raw_leaked)} chiếc lá nằm ở >1 phần, ví dụ {raw_leaked[:3]}")

    # Kiểm tra chéo theo bản đồ lá chính thức
    official: dict[str, set] = defaultdict(set)
    for r in plan["rows"]:
        if r.get("leaf_id"):
            official[r["leaf_id"]].add(r["split"])
    off_leaked = [k for k, s in official.items() if len(s) > 1]
    if off_leaked:
        errors.append(f"{len(off_leaked)} lá (leaf-map chính thức) nằm ở >1 phần, ví dụ {off_leaked[:3]}")

    for s in SPLITS:
        present = {r["class"] for r in plan["rows"] if r["split"] == s}
        missing = set(plan["classes"]) - present
        if missing:
            errors.append(f"Phần {s} thiếu lớp: {sorted(missing)}")

    if expected_classes is not None and plan["classes"] != expected_classes:
        errors.append("Danh sách/thứ tự lớp khác models/class_indices.json → sẽ lệch nhãn khi serve")
    return errors


def build_report(plan: dict) -> dict:
    per_class: dict[str, Counter] = defaultdict(Counter)
    for r in plan["rows"]:
        per_class[r["class"]][r["split"]] += 1
    totals = Counter(r["split"] for r in plan["rows"])
    n_groups = len({r["group"] for r in plan["rows"]})
    multi = Counter(r["group"] for r in plan["rows"])
    fingerprint = hashlib.sha256(
        "\n".join(f'{r["split"]}\t{r["class"]}/{r["file"]}'
                  for r in sorted(plan["rows"], key=lambda r: (r["class"], r["file"]))).encode()
    ).hexdigest()
    return {
        "source_files": plan["n_source_files"],
        "images_in_official_leaf_map": plan.get("n_in_leaf_map", 0),
        "kept_files": len(plan["rows"]),
        "dropped_byte_duplicates": plan["dropped_byte_duplicates"],
        "groups": n_groups,
        "groups_with_several_images": sum(1 for v in multi.values() if v > 1),
        "totals": {s: totals[s] for s in SPLITS},
        "per_class": {c: {s: per_class[c][s] for s in SPLITS} for c in plan["classes"]},
        "fractions": {"train": TRAIN_FRAC, "valid": VALID_FRAC, "test": round(1 - TRAIN_FRAC - VALID_FRAC, 2)},
        "split_fingerprint": fingerprint,
    }


# ----------------------------------------------------------------------------- ghi ra đĩa
def write_split(plan: dict, report: dict, dst: Path) -> None:
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"❌ {dst} đã có dữ liệu — xóa hoặc chọn --dst khác (không ghi đè).")
    for s in SPLITS:
        for c in plan["classes"]:
            (dst / s / c).mkdir(parents=True, exist_ok=True)
    for r in plan["rows"]:
        target = dst / r["split"] / r["class"] / r["file"]
        try:
            os.link(r["path"], target)        # hardlink: không tốn thêm ổ đĩa
        except OSError:
            shutil.copy2(r["path"], target)   # khác ổ đĩa / không hỗ trợ → copy
    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "file", "group", "official_leaf_id"])
        for r in plan["rows"]:
            w.writerow([r["split"], r["class"], r["file"], r["group"], r.get("leaf_id") or ""])
    with open(dst / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _expected_classes() -> list[str] | None:
    if not CLASS_INDICES.exists():
        return None
    c2i = json.loads(CLASS_INDICES.read_text(encoding="utf-8"))["class_to_idx"]
    return sorted(c2i, key=c2i.get)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="thư mục ảnh gốc (mỗi lớp 1 thư mục con)")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST, help="thư mục đích (phải trống)")
    ap.add_argument("--leaf-map", type=Path, default=DEFAULT_LEAF_MAP,
                    help="leaf-map.json chính thức của PlantVillage (mã chụp → mã lá)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in thống kê, không ghi gì")
    args = ap.parse_args(argv)

    if not args.src.is_dir():
        print(f"❌ Không thấy {args.src}", file=sys.stderr)
        return 1

    if not args.leaf_map.is_file():
        print(f"❌ Không thấy {args.leaf_map} — thiếu nó sẽ để lọt các tấm khác mã của cùng một lá.",
              file=sys.stderr)
        return 1
    leaf_map = json.loads(args.leaf_map.read_text(encoding="utf-8"))

    print(f"🔎 Đang đọc {args.src} …")
    plan = plan_split(args.src, leaf_map)
    errors = verify(plan, _expected_classes())
    report = build_report(plan)

    t = report["totals"]
    print(f"Ảnh nguồn: {report['source_files']} | giữ: {report['kept_files']} | "
          f"bỏ do trùng byte: {len(report['dropped_byte_duplicates'])}")
    print(f"Ảnh có trong leaf-map chính thức: {report['images_in_official_leaf_map']}")
    print(f"Số chiếc lá (nhóm): {report['groups']} | nhóm có >1 ảnh: {report['groups_with_several_images']}")
    print(f"train {t['train']} | valid {t['valid']} | test {t['test']}")
    smallest = min(report["per_class"].items(), key=lambda kv: kv[1]["test"])
    print(f"Lớp có ít ảnh test nhất: {smallest[0]} → {smallest[1]}")
    print(f"Dấu vân tay cách chia: {report['split_fingerprint'][:16]}…")

    if errors:
        print("❌ Kiểm tra thất bại, KHÔNG ghi gì:", *errors, sep="\n  - ", file=sys.stderr)
        return 2
    print("✅ Không có chiếc lá nào nằm ở 2 phần; đủ 38 lớp ở mỗi phần; thứ tự lớp khớp model.")

    if args.dry_run:
        print("(dry-run) Không ghi gì.")
        return 0
    write_split(plan, report, args.dst)
    print(f"💾 Đã ghi {args.dst} (+ manifest.csv, split_report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
