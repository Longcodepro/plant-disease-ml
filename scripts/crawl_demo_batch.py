#!/usr/bin/env python3
"""
Script cào ảnh lá cây thực tế từ Wikimedia Commons & iNaturalist (API công khai, không cần API key)
để test nhanh tính năng "Chẩn đoán hàng loạt (Tối đa 32 ảnh)" trên Web UI.

Cách dùng:
    python scripts/crawl_demo_batch.py
    python scripts/crawl_demo_batch.py --out-dir ~/Downloads/plant_batch_demo --limit 30
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

USER_AGENT = "PlantGuardDemoCrawler/1.0 (educational ML project)"

# (nhãn_chuẩn_PlantVillage, từ_khóa_tìm_kiếm)
DEFAULT_QUERIES: List[Tuple[str, str]] = [
    ("Apple___Apple_scab", "Venturia inaequalis apple scab leaf"),
    ("Apple___Black_rot", "Botryosphaeria obtusa apple leaf"),
    ("Apple___Cedar_apple_rust", "Gymnosporangium juniperi-virginianae apple leaf"),
    ("Apple___healthy", "Malus domestica green leaf"),
    ("Cherry_(including_sour)___Powdery_mildew", "Podosphaera clandestina cherry leaf"),
    ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Cercospora zeae-maydis corn leaf"),
    ("Corn_(maize)___Common_rust_", "Puccinia sorghi corn leaf"),
    ("Corn_(maize)___Northern_Leaf_Blight", "Exserohilum turcicum corn leaf"),
    ("Grape___Black_rot", "Guignardia bidwellii grape leaf"),
    ("Grape___Esca_(Black_Measles)", "grapevine esca leaf symptoms"),
    ("Grape___healthy", "Vitis vinifera green leaf"),
    ("Orange___Haunglongbing_(Citrus_greening)", "citrus greening huanglongbing leaf"),
    ("Peach___Bacterial_spot", "Xanthomonas arboricola pruni peach leaf"),
    ("Pepper,_bell___Bacterial_spot", "Xanthomonas campestris pepper leaf"),
    ("Potato___Early_blight", "Alternaria solani potato leaf"),
    ("Potato___Late_blight", "Phytophthora infestans potato leaf"),
    ("Squash___Powdery_mildew", "Podosphaera xanthii squash leaf"),
    ("Strawberry___Leaf_scorch", "Diplocarpon earlianum strawberry leaf"),
    ("Tomato___Bacterial_spot", "tomato bacterial spot leaf"),
    ("Tomato___Early_blight", "Alternaria solani tomato leaf"),
    ("Tomato___Late_blight", "Phytophthora infestans tomato leaf"),
    ("Tomato___Leaf_Mold", "Passalora fulva tomato leaf"),
    ("Tomato___Septoria_leaf_spot", "Septoria lycopersici tomato leaf"),
    ("Tomato___Spider_mites Two-spotted_spider_mite", "Tetranychus urticae tomato leaf"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato yellow leaf curl virus leaf"),
    ("Tomato___Tomato_mosaic_virus", "tomato mosaic virus leaf"),
    ("Tomato___healthy", "Solanum lycopersicum healthy leaf"),
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


def _is_valid_image_bytes(data: bytes) -> bool:
    return (
        len(data) >= 4096
        and (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n"))
    )


def search_wikimedia(query: str, ctx: ssl.SSLContext, limit: int = 4) -> List[str]:
    """Tìm URL ảnh JPEG/PNG (cỡ ~640px) qua Wikimedia Commons API."""
    params = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
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


def search_inaturalist(query: str, ctx: ssl.SSLContext, limit: int = 3) -> List[str]:
    """Fallback sang iNaturalist API để lấy ảnh chụp thực tế ngoài tự nhiên."""
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
    limit: int = 30,
    per_class: int = 1,
) -> List[Path]:
    """Cào tối đa `limit` ảnh thực tế từ Internet vào `out_dir` để test chẩn đoán hàng loạt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context()
    saved: List[Path] = []
    seen_urls = set()

    for label, query in queries:
        if len(saved) >= limit:
            break
        candidates: List[str] = []
        try:
            candidates.extend(search_wikimedia(query, ctx, limit=4))
        except Exception as e:
            print(f"⚠️  Wikimedia lỗi '{query}': {e}")
        if len(candidates) < per_class:
            try:
                candidates.extend(search_inaturalist(query, ctx, limit=3))
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
                if not _is_valid_image_bytes(raw):
                    continue
                ext = ".png" if raw.startswith(b"\x89PNG") else ".jpg"
                safe_label = label.replace(",", "").replace(" ", "_")
                dst = out_dir / f"{len(saved) + 1:02d}_{safe_label}{ext}"
                dst.write_bytes(raw)
                saved.append(dst)
                got += 1
                print(f"  ✅ [{len(saved):02d}/{limit}] {dst.name} ({len(raw)/1024:.0f} KB)")
            except Exception:
                continue

    print(f"\n🎉 Đã cào xong {len(saved)} ảnh vào: {out_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Cào ảnh thực tế Internet test Batch Prediction")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "Downloads" / "plant_batch_demo")
    parser.add_argument("--limit", type=int, default=30, help="Tổng số ảnh tối đa (Batch API giới hạn 32)")
    parser.add_argument("--per-class", type=int, default=1, help="Số ảnh mỗi lớp bệnh")
    args = parser.parse_args()
    crawl_batch_images(out_dir=args.out_dir, limit=min(args.limit, 32), per_class=args.per_class)


if __name__ == "__main__":
    main()
