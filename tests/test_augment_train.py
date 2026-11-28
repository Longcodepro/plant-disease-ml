"""
Test cho scripts/augment_train.py (cần Pillow).

Chạy:
    python3 -m unittest tests/test_augment_train.py -v
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import augment_train as aug  # noqa: E402


def _img(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


class AugmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src"
        for i in range(3):
            _img(self.src / "train" / "Small" / f"s{i}.JPG", (i * 40, 120, 30))
        for i in range(6):
            _img(self.src / "train" / "Big" / f"b{i}.JPG", (200, i * 30, 30))
        for split in ("valid", "test"):
            for c in ("Small", "Big"):
                _img(self.src / split / c / f"{split}_{c}.JPG", (10, 10, 10))

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_train_is_augmented_and_classes_balanced(self):
        dst = Path(self.tmp.name) / "out"
        self.assertEqual(aug.main(["--src", str(self.src), "--dst", str(dst), "--target", "5", "--workers", "1"]), 0)
        self.assertEqual(len(list((dst / "train" / "Small").iterdir())), 5)   # 3 gốc + 2 sinh thêm
        self.assertEqual(len(list((dst / "train" / "Big").iterdir())), 6)     # đã đủ → giữ nguyên
        for split in ("valid", "test"):
            names = [p.name for p in (dst / split).rglob("*.JPG")]
            self.assertEqual(len(names), 2)
            self.assertFalse(any(aug.AUG_TAG in n for n in names))
        with open(dst / "augment_manifest.csv") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        for r in rows:  # mọi ảnh sinh ra đều xuất phát từ ảnh TRAIN cùng lớp
            self.assertTrue((self.src / "train" / r["class"] / r["source_train_file"]).exists())

    def test_deterministic(self):
        im = Image.open(self.src / "train" / "Small" / "s1.JPG").convert("RGB")
        a, _ = aug.augment_image(im, aug._rng("Small", "s1.JPG", 1))
        b, _ = aug.augment_image(im, aug._rng("Small", "s1.JPG", 1))
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(a.size, im.size)


if __name__ == "__main__":
    unittest.main()
