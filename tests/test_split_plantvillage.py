"""
Test cho scripts/split_plantvillage.py — chỉ dùng thư viện chuẩn.

Chạy:
    python3 -m unittest tests/test_split_plantvillage.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import split_plantvillage as sp  # noqa: E402

U1 = "00075aa8-d81a-4184-8541-b692b78d398a"
U2 = "01a66316-0e98-4d3b-a56f-d78752cd043f"


class LeafKeyTest(unittest.TestCase):
    def test_strips_uuid_extension_and_case(self):
        self.assertEqual(sp.leaf_key("A", f"{U1}___FREC_Scab 3335.JPG"), "A/frec_scab 3335")

    def test_same_leaf_different_days_share_key(self):
        a = sp.leaf_key("T", f"{U1}___GHLB Leaf 1.1 Day 6.JPG")
        b = sp.leaf_key("T", f"{U2}___GHLB Leaf 1.1 Day 12.jpg")
        self.assertEqual(a, b)

    def test_different_leaves_differ(self):
        self.assertNotEqual(sp.leaf_key("T", f"{U1}___GHLB Leaf 1.1 Day 6.JPG"),
                            sp.leaf_key("T", f"{U1}___GHLB Leaf 1.2 Day 6.JPG"))

    def test_same_photo_id_in_different_class_is_not_merged(self):
        self.assertNotEqual(sp.leaf_key("A", "x___RS_HL 1.JPG"), sp.leaf_key("B", "x___RS_HL 1.JPG"))


class SplitOfTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(sp.split_of("abc/def"), sp.split_of("abc/def"))

    def test_proportions_roughly_70_15_15(self):
        from collections import Counter
        c = Counter(sp.split_of(f"k{i}") for i in range(20000))
        self.assertAlmostEqual(c["train"] / 20000, 0.70, delta=0.02)
        self.assertAlmostEqual(c["valid"] / 20000, 0.15, delta=0.02)
        self.assertAlmostEqual(c["test"] / 20000, 0.15, delta=0.02)


class PlanTest(unittest.TestCase):
    def _make(self, root: Path, files: dict[str, bytes]) -> None:
        for rel, data in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def test_groups_never_split_and_byte_duplicates_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            files = {}
            for cls in ("A", "B"):
                for i in range(60):
                    files[f"{cls}/{U1[:-4]}{i:04d}___P Leaf {i} Day 1.JPG"] = f"{cls}{i}d1".encode()
                    files[f"{cls}/{U2[:-4]}{i:04d}___P Leaf {i} Day 9.JPG"] = f"{cls}{i}d9".encode()
            files["A/zzz___copy.JPG"] = b"A0d1"  # trùng byte với một ảnh khác
            self._make(src, files)

            plan = sp.plan_split(src)
            self.assertEqual(sp.verify(plan, None), [])
            self.assertEqual(len(plan["dropped_byte_duplicates"]), 1)
            self.assertEqual(len(plan["rows"]), 240)
            # cùng chiếc lá (Day 1 & Day 9) → cùng phần
            by_leaf = {}
            for r in plan["rows"]:
                by_leaf.setdefault(sp.leaf_key(r["class"], r["file"]), set()).add(r["split"])
            self.assertTrue(all(len(s) == 1 for s in by_leaf.values()))

            dst = Path(d) / "out"
            sp.write_split(plan, sp.build_report(plan), dst)
            n = sum(1 for s in sp.SPLITS for _ in (dst / s).rglob("*.JPG"))
            self.assertEqual(n, 240)
            rep = json.loads((dst / "split_report.json").read_text())
            self.assertEqual(sum(rep["totals"].values()), 240)
            with self.assertRaises(SystemExit):  # không ghi đè
                sp.write_split(plan, sp.build_report(plan), dst)

    def test_official_leaf_map_keeps_different_photo_ids_of_one_leaf_together(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            files, leaf_map = {}, {}
            for cls in ("A", "B"):
                for leaf in range(40):
                    for shot in range(3):  # 3 mã chụp khác nhau, cùng một lá
                        pid = f"rs {cls} {leaf * 10 + shot}"
                        files[f"{cls}/{U1}___{pid.upper()}.JPG"] = f"{cls}{leaf}{shot}".encode()
                        leaf_map[pid.lower()] = [f"{cls}:::{leaf}.0"]
            self._make(src, files)
            plan = sp.plan_split(src, leaf_map)
            self.assertEqual(sp.verify(plan, None), [])
            self.assertEqual(plan["n_in_leaf_map"], 240)
            splits_per_leaf = {}
            for r in plan["rows"]:
                splits_per_leaf.setdefault(r["leaf_id"], set()).add(r["split"])
            self.assertEqual(len(splits_per_leaf), 80)
            self.assertTrue(all(len(s) == 1 for s in splits_per_leaf.values()))

    def test_verify_detects_wrong_class_order(self):
        plan = {"classes": ["A", "B"], "rows": [
            {"split": s, "class": c, "file": f"{c}{s}.JPG", "group": f"{c}/{s}"}
            for c in "AB" for s in sp.SPLITS]}
        self.assertEqual(sp.verify(plan, ["A", "B"]), [])
        self.assertTrue(sp.verify(plan, ["B", "A"]))

    def test_verify_detects_leak(self):
        plan = {"classes": ["A"], "rows": [
            {"split": "train", "class": "A", "file": "x___L 1 Day 1.JPG", "group": "g1"},
            {"split": "valid", "class": "A", "file": "y___L 1 Day 2.JPG", "group": "g2"},
            {"split": "test", "class": "A", "file": "z___L 2.JPG", "group": "g3"}]}
        self.assertTrue(any("lá" in e for e in sp.verify(plan, None)))


if __name__ == "__main__":
    unittest.main()
