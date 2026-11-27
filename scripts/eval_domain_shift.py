#!/usr/bin/env python3
"""
Đo mức độ Domain Shift (Lab -> Thực tế ngoài đồng ruộng) trên tập PlantDoc Test.
Chạy trực tiếp file `.onnx` (không cần PyTorch) và xuất báo cáo JSON.

Cách chạy:
    python scripts/eval_domain_shift.py --model models/convnext_tiny_khoi_int8.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[None, ...]


def evaluate_on_plantdoc(
    model_path: Path,
    plantdoc_test_dir: Path,
    mapping_path: Path,
    out_json: Path | None = None,
) -> dict:
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    y_true: list[str] = []
    y_pred: list[str] = []
    top3_hits = 0
    corrupted = 0
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    latencies_ms: list[float] = []

    for cls_dir in sorted(p for p in plantdoc_test_dir.iterdir() if p.is_dir()):
        gt_class = cls_dir.name
        if gt_class not in class_to_idx:
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file():
                continue
            try:
                x = preprocess_image(img_path)
            except Exception:
                corrupted += 1
                continue

            t0 = time.perf_counter()
            logits = sess.run(None, {input_name: x})[0][0]
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            top3_idx = np.argsort(logits)[::-1][:3]
            pred_class = idx_to_class[int(top3_idx[0])]
            top3_classes = [idx_to_class[int(i)] for i in top3_idx]

            y_true.append(gt_class)
            y_pred.append(pred_class)
            per_class_total[gt_class] += 1
            if pred_class == gt_class:
                per_class_correct[gt_class] += 1
            else:
                confusion_pairs[(gt_class, pred_class)] += 1
            if gt_class in top3_classes:
                top3_hits += 1

    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / total if total else 0.0
    top3_acc = top3_hits / total if total else 0.0

    # Macro-F1 trên các lớp có mặt trong PlantDoc Test
    present_classes = sorted(per_class_total.keys())
    f1s = []
    per_class_report = []
    for cls in present_classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        per_class_report.append({
            "class_name": cls,
            "samples": per_class_total[cls],
            "correct": per_class_correct[cls],
            "accuracy": round(rec, 4),
            "f1": round(f1, 4),
        })

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    report = {
        "model": str(model_path),
        "dataset": "PlantDoc Test (In-the-Wild)",
        "num_classes_evaluated": len(present_classes),
        "total_images": total,
        "corrupted_skipped": corrupted,
        "top1_accuracy": round(acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in confusion_pairs.most_common(10)
        ],
        "per_class": sorted(per_class_report, key=lambda r: r["accuracy"], reverse=True),
    }

    print("\n==================================================================")
    print(f" 🌍 DOMAIN SHIFT BENCHMARK — PLANTDOC TEST ({total} ảnh ngoài trời)")
    print("==================================================================")
    print(f" 📦 Model             : {model_path.name}")
    print(f" 🏷️  Số lớp đánh giá   : {len(present_classes)} lớp chung với PlantVillage")
    print(f" 🎯 Top-1 Accuracy    : {acc * 100:.2f}% ({correct}/{total})")
    print(f" 🥈 Top-3 Accuracy    : {top3_acc * 100:.2f}% ({top3_hits}/{total})")
    print(f" 📊 Macro F1-Score    : {macro_f1 * 100:.2f}%")
    print(f" ⚡ Avg CPU Latency   : {report['avg_latency_ms']:.2f} ms/ảnh")
    print("------------------------------------------------------------------")
    print(" 🔁 Top 5 cặp nhầm lẫn phổ biến nhất:")
    for item in report["top_confusions"][:5]:
        print(f"   • {item['true']} -> {item['predicted']} ({item['count']} ảnh)")
    print("==================================================================\n")

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"💾 Đã lưu báo cáo Domain Shift tại: {out_json}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá Domain Shift trên PlantDoc Test")
    parser.add_argument("--model", type=Path, default=Path("models/convnext_tiny_khoi_int8.onnx"))
    default_data = Path("data_plantdoc/test") if Path("data_plantdoc/test").exists() else Path("data/plantdoc/test")
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--mapping", type=Path, default=Path("models/class_indices.json"))
    parser.add_argument("--out", type=Path, default=Path("notebooks/eda_reports/plantdoc_domain_shift.json"))
    args = parser.parse_args()
    evaluate_on_plantdoc(args.model, args.data, args.mapping, args.out)
