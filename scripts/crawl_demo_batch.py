#!/usr/bin/env python3
"""
Script cào ảnh lá cây thực tế từ Wikimedia Commons & iNaturalist (API công khai, không cần API key),
tự động lọc bỏ ảnh kính hiển vi / đen trắng và kiểm chứng bằng model ONNX (nếu có)
để tạo bộ ảnh sạch test "Chẩn đoán hàng loạt (Tối đa 32 ảnh)" trên Web UI.

Cách dùng:
    python scripts/crawl_demo_batch.py
    python scripts/crawl_demo_batch.py --out-dir ~/Downloads/plant_batch_demo --limit 25
"""

from __future__ import annotations

import argparse
import io
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

USER_AGENT = "PlantGuardDemoCrawler/1.1 (educational ML project)"
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dùng từ khóa ảnh chụp lá cây thông dụng (tránh tên khoa học nấm dễ ra ảnh kính hiển vi SEM)
DEFAULT_QUERIES: List[Tuple[str, str]] = [
    ("Apple___Apple_scab", "apple scab on leaf"),
    ("Apple___Black_rot", "apple black rot leaf spot"),
    ("Apple___Cedar_apple_rust", "cedar apple rust apple leaf"),
    ("Apple___healthy", "apple tree leaf green"),
    ("Cherry_(including_sour)___Powdery_mildew", "cherry powdery mildew leaf"),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "gray leaf spot corn leaf"),
    ("Corn_(maize)___Common_rust_", "corn common rust leaf"),
    ("Corn_(maize)___Northern_Leaf_Blight", "northern corn leaf blight"),
    ("Grape___Black_rot", "grape black rot leaf"),
    ("Grape___Esca_(Black_Measles)", "grapevine esca leaf"),
    ("Grape___healthy", "grapevine leaf green"),
    ("Orange___Haunglongbing_(Citrus_greening)", "citrus greening orange leaf"),
    ("Peach___Bacterial_spot", "peach bacterial spot leaf"),
    ("Pepper,_bell___Bacterial_spot", "pepper bacterial leaf spot"),
    ("Potato___Early_blight", "potato early blight leaf"),
    ("Potato___Late_blight", "potato late blight leaf"),
    ("Squash___Powdery_mildew", "squash powdery mildew leaf"),
    ("Strawberry___Leaf_scorch", "strawberry leaf scorch"),
    ("Tomato___Bacterial_spot", "tomato bacterial spot leaf"),
    ("Tomato___Early_blight", "tomato early blight leaf"),
    ("Tomato___Late_blight", "tomato late blight leaf"),
    ("Tomato___Leaf_Mold", "tomato leaf mold"),
    ("Tomato___Septoria_leaf_spot", "tomato septoria leaf spot"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato yellow leaf curl virus"),
    ("Tomato___Tomato_mosaic_virus", "tomato mosaic virus leaf"),
    ("Tomato___healthy", "tomato plant leaf green"),
]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch_bytes(url: str, ctx: ssl.SSLContext, timeout: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read()


def _is_color_leaf_photo(data: bytes) -> Optional[Image.Image]:
    """Loại file hỏng, ảnh quá nhỏ (<200px), banner dài và ảnh kính hiển vi đen trắng."""
    if len(data) < 4096:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    if min(w, h) < 200 or max(w, h) / max(1, min(w, h)) > 2.5:
        return None
    arr = np.asarray(img.resize((64, 64)), dtype=np.float32)
    # Ảnh SEM/kính hiển vi đen trắng có độ lệch giữa các kênh R,G,B rất thấp
    channel_spread = float(np.mean(np.max(arr, axis=2) - np.min(arr, axis=2)))
    if channel_spread < 12.0:
        return None
    return img


def _load_verifier(model_path: Path, mapping_path: Path):
    if not model_path.exists() or not mapping_path.exists():
        return None
    try:
        import onnxruntime as ort
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_to_idx = mapping["class_to_idx"]
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def check_top3(img: Image.Image, target_label: str) -> bool:
            target_idx = class_to_idx.get(target_label)
            if target_idx is None:
                return True
            arr = (np.asarray(img.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0 - mean) / std
            logits = sess.run(None, {inp: np.transpose(arr, (2, 0, 1))[None, ...]})[0][0]
            top3 = np.argsort(logits)[::-1][:3]
            return int(target_idx) in [int(i) for i in top3]

        return check_top3
    except Exception:
        return None


def search_wikimedia(query: str, ctx: ssl.SSLContext, limit: int = 8) -> List[str]:
    """Tìm URL ảnh JPEG/PNG (cỡ ~640px) qua Wikimedia Commons API."""
    params = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": "640",
        "format": "json",
    })
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    data = json.loads(_fetch_bytes(url, ctx).decode("utf-8"))
    pages: Dict[str, dict] = data.get("query", {}).get("pages", {})
    urls: List[str] = []
    for page in pages.values():
        title = (page.get("title") or "").lower()
        if any(bad in title for bad in ("sem", "microscop", "spore", "diagram", "map", "chart", "gall")):
            continue
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        img_url = info.get("thumburl") or info.get("url")
        if img_url:
            urls.append(img_url)
    return urls


def search_inaturalist(query: str, ctx: ssl.SSLContext, limit: int = 6) -> List[str]:
    """Tìm ảnh chụp thực tế ngoài tự nhiên qua iNaturalist API."""
    params = urllib.parse.urlencode({"q": query, "photos": "true", "per_page": str(limit)})
    url = f"https://api.inaturalist.org/v1/observations?{params}"
    data = json.loads(_fetch_bytes(url, ctx).decode("utf-8"))
    urls: List[str] = []
    for obs in data.get("results", []):
        for photo in obs.get("photos", []):
            raw_url = photo.get("url")
            if raw_url:
                urls.append(raw_url.replace("square.", "medium."))
                break
    return urls


def crawl_batch_images(
    out_dir: Path,
    queries: List[Tuple[str, str]] = DEFAULT_QUERIES,
    limit: int = 25,
    per_class: int = 1,
    model_path: Path = ROOT_DIR / "models" / "convnext_tiny_field_v2_int8.onnx",
    mapping_path: Path = ROOT_DIR / "models" / "class_indices.json",
) -> List[Path]:
    """Cào tối đa `limit` ảnh lá cây thực tế sạch từ Internet vào `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context()
    verifier = _load_verifier(model_path, mapping_path)
    saved: List[Path] = []
    seen_urls = set()

    for label, query in queries:
        if len(saved) >= limit:
            break
        candidates: List[str] = []
        try:
            time.sleep(0.5)
            candidates.extend(search_wikimedia(query, ctx, limit=8))
        except Exception as e:
            print(f"⚠️  Wikimedia lỗi '{query}': {e}")
        try:
            time.sleep(0.3)
            candidates.extend(search_inaturalist(query, ctx, limit=6))
        except Exception as e:
            print(f"⚠️  iNaturalist lỗi '{query}': {e}")

        got = 0
        for img_url in candidates:
            if got >= per_class or len(saved) >= limit:
                break
            if img_url in seen_urls:
                continue
            seen_urls.add(img_url)
            try:
                raw = _fetch_bytes(img_url, ctx)
                img = _is_color_leaf_photo(raw)
                if img is None:
                    continue
                if verifier is not None and not verifier(img, label):
                    continue
                if max(img.size) > 800:
                    img.thumbnail((800, 800), Image.LANCZOS)
                safe_label = label.replace(",", "").replace(" ", "_")
                dst = out_dir / f"{len(saved) + 1:02d}_{safe_label}.jpg"
                img.save(dst, format="JPEG", quality=90)
                saved.append(dst)
                got += 1
                print(f"  ✅ [{len(saved):02d}/{limit}] {dst.name} ({dst.stat().st_size/1024:.0f} KB)")
            except Exception:
                continue

    print(f"\n🎉 Đã cào & lọc xong {len(saved)} ảnh lá sạch vào: {out_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Cào ảnh thực tế Internet test Batch Prediction")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "Downloads" / "plant_batch_demo")
    parser.add_argument("--limit", type=int, default=25, help="Tổng số ảnh tối đa (Batch API giới hạn 32)")
    parser.add_argument("--per-class", type=int, default=1, help="Số ảnh mỗi lớp bệnh")
    parser.add_argument("--model", type=Path, default=ROOT_DIR / "models" / "convnext_tiny_field_v2_int8.onnx")
    args = parser.parse_args()
    crawl_batch_images(out_dir=args.out_dir, limit=min(args.limit, 32), per_class=args.per_class, model_path=args.model)


if __name__ == "__main__":
    main()
