"""
Streamlit Web UI: Chẩn đoán Bệnh Lá Cây Nông Nghiệp (Tier 3)
Kết nối trực tiếp tới Backend API (FastAPI hoặc Rust API trên cùng hợp đồng HTTP):
  - GET  /health
  - GET  /metrics
  - POST /predict
  - POST /predict/batch
  - POST /explain
"""

from __future__ import annotations

import base64
import csv
import io
import os
from pathlib import Path
from typing import Tuple

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Plant Disease AI — Chẩn đoán Bệnh Lá Cây",
    page_icon="🌿",
    layout="wide",
)

# Custom CSS nhẹ, sạch mắt, tương thích cả Light & Dark theme
st.markdown(
    """
    <style>
    .status-card {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        border-left: 6px solid #2e7d32;
        background: rgba(46, 125, 50, 0.10);
        margin-bottom: 1rem;
    }
    .status-card.disease {
        border-left-color: #d84315;
        background: rgba(216, 67, 21, 0.10);
    }
    .status-card.uncertain {
        border-left-color: #f9a825;
        background: rgba(249, 168, 37, 0.12);
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        background: rgba(25, 118, 210, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT_DIR / "data" / "test"
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def format_label(raw_label: str) -> Tuple[str, str, bool]:
    """Tách nhãn PlantVillage 'Apple___Cedar_apple_rust' -> ('Apple', 'Cedar apple rust', False)."""
    if "___" in raw_label:
        plant, cond = raw_label.split("___", 1)
    else:
        plant, cond = "Cây trồng", raw_label
    plant_clean = plant.replace("_", " ").replace(",", ", ").strip()
    cond_clean = cond.replace("_", " ").strip()
    is_healthy = "healthy" in cond_clean.lower()
    return plant_clean, ("Khỏe mạnh (Healthy)" if is_healthy else cond_clean), is_healthy


def load_sample_files() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(
        [p for p in SAMPLE_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )


# ── Sidebar: Cấu hình & Trạng thái Backend ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Kết nối Backend API")
    api_base = st.text_input("Địa chỉ API", value=DEFAULT_API_URL, help="Hỗ trợ cả FastAPI lẫn Rust API").rstrip("/")

    api_healthy = False
    try:
        resp = requests.get(f"{api_base}/health", timeout=2)
        if resp.status_code == 200:
            h = resp.json()
            api_healthy = bool(h.get("model_loaded"))
            st.success("🟢 Backend API: Sẵn sàng")
            st.info(f"📦 Mô hình: `{h.get('model_version', 'unknown')}`")
        else:
            st.warning(f"🟡 Backend trả mã {resp.status_code}")
    except Exception:
        st.error("🔴 Chưa kết nối được Backend API")
        st.caption("Chạy `uvicorn backend.main:app --port 8000` hoặc Rust API.")

    st.markdown("---")
    st.caption("Kiến trúc 3 tầng: ConvNeXt-Tiny ONNX INT8 (99.95% Val F1 · 23.8 ms trên M2)")


st.title("🌿 Hệ Thống Chẩn Đoán Bệnh Trên Lá Cây Nông Nghiệp")
st.caption("Nhận diện 38 loại cây trồng & bệnh lý (PlantVillage) · Giải thích Heatmap XAI · Giám sát trực tuyến")

tab_single, tab_batch, tab_monitor = st.tabs(
    ["🔍 Chẩn đoán & Giải thích XAI", "📦 Chẩn đoán hàng loạt", "📊 Giám sát hệ thống"]
)

# ==============================================================================
# TAB 1: CHẨN ĐOÁN 1 ẢNH + HEATMAP XAI
# ==============================================================================
with tab_single:
    col_in, col_out = st.columns([1, 1.15], gap="large")

    img_bytes: bytes | None = None
    img_name = "leaf.jpg"
    img_mime = "image/jpeg"

    with col_in:
        st.subheader("1. Chọn ảnh lá cây")
        samples = load_sample_files()
        source_mode = (
            st.radio("Nguồn ảnh", ["Tải ảnh từ máy", "Ảnh mẫu có sẵn (data/test)"], horizontal=True)
            if samples
            else "Tải ảnh từ máy"
        )

        if source_mode == "Tải ảnh từ máy":
            uploaded = st.file_uploader("Chọn ảnh lá cây (JPG, PNG)...", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                img_bytes = uploaded.getvalue()
                img_name = uploaded.name
                img_mime = uploaded.type or "image/jpeg"
        else:
            chosen_name = st.selectbox("Chọn 1 ảnh kiểm thử độc lập", [p.name for p in samples])
            chosen_path = SAMPLE_DIR / chosen_name
            img_bytes = chosen_path.read_bytes()
            img_name = chosen_path.name
            img_mime = "image/png" if chosen_path.suffix.lower() == ".png" else "image/jpeg"

        if img_bytes is not None:
            st.image(Image.open(io.BytesIO(img_bytes)), caption=f"Ảnh đầu vào: {img_name}", use_container_width=True)

    with col_out:
        st.subheader("2. Kết quả chẩn đoán")
        if img_bytes is not None:
            show_xai = st.checkbox("🔥 Kèm giải thích Heatmap XAI (khoanh đỏ vùng vết bệnh)", value=True)
            if st.button("🔍 Tiến hành Chẩn đoán", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích hình ảnh..."):
                    try:
                        files = {"file": (img_name, img_bytes, img_mime)}
                        pred_resp = requests.post(f"{api_base}/predict", files=files, timeout=15)
                        if pred_resp.status_code != 200:
                            st.error(f"Lỗi từ API ({pred_resp.status_code}): {pred_resp.text}")
                        else:
                            res = pred_resp.json()
                            plant, condition, is_healthy = format_label(res["predicted_class"])
                            is_uncertain = bool(res.get("is_uncertain", False))
                            card_cls = "uncertain" if is_uncertain else ("" if is_healthy else "disease")
                            icon = "⚠️" if is_uncertain else ("✅" if is_healthy else "🩺")

                            st.markdown(
                                f"""
                                <div class="status-card {card_cls}">
                                    <div style="font-size:0.9rem; opacity:0.85;">🌱 Loại cây: <b>{plant}</b></div>
                                    <div style="font-size:1.35rem; font-weight:700; margin:0.25rem 0;">{icon} {condition}</div>
                                    <div>
                                        <span class="badge">🎯 Độ tin cậy: {res['confidence']*100:.2f}%</span>
                                        <span class="badge">⏱️ {res['inference_time_ms']} ms</span>
                                        {'<span class="badge">⚡ Cache Hit</span>' if res.get('cache_hit') else ''}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if res.get("warning"):
                                st.warning(res["warning"])

                            st.markdown("**Top 3 khả năng cao nhất:**")
                            for item in res.get("top_3", []):
                                p_name, c_name, _ = format_label(item["class_name"])
                                conf = float(item["confidence"])
                                st.progress(min(max(conf, 0.0), 1.0))
                                st.caption(f"**{p_name} — {c_name}**: `{conf*100:.2f}%`")

                            if show_xai:
                                with st.spinner("Đang sinh bản đồ nhiệt XAI (Occlusion Sensitivity)..."):
                                    ex_resp = requests.post(
                                        f"{api_base}/explain",
                                        files={"file": (img_name, img_bytes, img_mime)},
                                        timeout=30,
                                    )
                                if ex_resp.status_code == 200:
                                    ex = ex_resp.json()
                                    heat_img = Image.open(io.BytesIO(base64.b64decode(ex["heatmap_base64"])))
                                    st.markdown("---")
                                    st.markdown(f"**🔥 Bản đồ nhiệt XAI** (`{ex['method']}` · `{ex['explain_time_ms']} ms`)")
                                    c1, c2 = st.columns(2)
                                    c1.image(Image.open(io.BytesIO(img_bytes)), caption="Ảnh gốc", use_container_width=True)
                                    c2.image(heat_img, caption="Vùng đỏ/vàng = nơi mô hình tập trung", use_container_width=True)
                                else:
                                    st.info("Endpoint `/explain` chưa khả dụng trên backend hiện tại (ví dụ Rust API).")
                    except Exception as e:
                        st.error(f"Không thể kết nối đến API: {e}")
        else:
            st.info("👈 Chọn ảnh mẫu hoặc tải lên ảnh lá cây ở cột bên trái để bắt đầu.")


# ==============================================================================
# TAB 2: CHẨN ĐOÁN HÀNG LOẠT (/predict/batch)
# ==============================================================================
with tab_batch:
    st.subheader("Chẩn đoán nhiều ảnh trong 1 batch ONNX duy nhất")
    batch_files = st.file_uploader(
        "Chọn tối đa 32 ảnh lá cây (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.caption(f"Đã chọn **{len(batch_files)}** ảnh.")
        if st.button("🚀 Chạy suy luận hàng loạt", type="primary"):
            with st.spinner(f"Đang xử lý batch {len(batch_files)} ảnh..."):
                try:
                    payload = [
                        ("files", (f.name, f.getvalue(), f.type or "image/jpeg"))
                        for f in batch_files[:32]
                    ]
                    resp = requests.post(f"{api_base}/predict/batch", files=payload, timeout=60)
                    if resp.status_code != 200:
                        st.error(f"Lỗi batch API ({resp.status_code}): {resp.text}")
                    else:
                        data = resp.json()
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Số ảnh xử lý", data["count"])
                        m2.metric("Tổng thời gian batch", f"{data['total_inference_time_ms']} ms")
                        avg_ms = data["total_inference_time_ms"] / max(1, data["count"])
                        m3.metric("Trung bình / ảnh", f"{avg_ms:.1f} ms")

                        rows = []
                        cols = st.columns(4)
                        for idx, (f, item) in enumerate(zip(batch_files[:32], data["results"])):
                            plant, cond, is_healthy = format_label(item["predicted_class"])
                            rows.append(
                                {
                                    "file": item["filename"],
                                    "plant": plant,
                                    "condition": cond,
                                    "confidence": f"{item['confidence']*100:.2f}%",
                                    "uncertain": "Có" if item.get("is_uncertain") else "Không",
                                }
                            )
                            with cols[idx % 4]:
                                st.image(Image.open(io.BytesIO(f.getvalue())), use_container_width=True)
                                icon = "✅" if is_healthy else "🩺"
                                st.caption(f"{icon} **{cond}** (`{item['confidence']*100:.1f}%`)")

                        st.dataframe(rows, use_container_width=True)

                        # Xuất CSV không cần phụ thuộc pandas
                        csv_buf = io.StringIO()
                        writer = csv.DictWriter(csv_buf, fieldnames=["file", "plant", "condition", "confidence", "uncertain"])
                        writer.writeheader()
                        writer.writerows(rows)
                        st.download_button(
                            "⬇️ Tải kết quả CSV",
                            data=csv_buf.getvalue().encode("utf-8-sig"),
                            file_name="plant_disease_batch_results.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Lỗi kết nối `/predict/batch`: {e}")


# ==============================================================================
# TAB 3: LIVE MONITORING (/metrics)
# ==============================================================================
with tab_monitor:
    col_title, col_btn = st.columns([4, 1])
    col_title.subheader("Giám sát hiệu năng & tín hiệu Data Drift trực tuyến")
    col_btn.button("🔄 Làm mới")

    try:
        m_resp = requests.get(f"{api_base}/metrics", timeout=5)
        if m_resp.status_code == 200:
            m = m_resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng Requests", m["total_requests"], f"{m['total_images_predicted']} ảnh")
            c2.metric("Latency p50 / p95", f"{m['latency_ms']['p50']} ms", f"p95: {m['latency_ms']['p95']} ms")
            c3.metric("Cache Hit Rate", f"{m['cache_hit_rate']*100:.1f}%", f"{m['cache_hits']} hits")
            c4.metric("Uncertain / Drift Rate", f"{m['uncertain_rate']*100:.1f}%", f"{m['uncertain_count']} ảnh")

            st.caption(
                f"Uptime: `{m['uptime_seconds']}s` · Backend: `{m['backend_type']}` · Model: `{m['model_version']}`"
            )

            top_cls = m.get("top_predicted_classes") or {}
            if top_cls:
                st.markdown("**Top lớp bệnh được chẩn đoán nhiều nhất:**")
                pretty_counts = {
                    f"{format_label(k)[0]} - {format_label(k)[1]}": v for k, v in top_cls.items()
                }
                st.bar_chart(pretty_counts)
            else:
                st.info("Chưa có lượt dự đoán nào — hãy thử chẩn đoán vài ảnh ở Tab 1 rồi quay lại xem số liệu.")
        else:
            st.warning(f"Backend không trả về `/metrics` (HTTP {m_resp.status_code}).")
    except Exception as e:
        st.error(f"Không thể lấy metrics từ `{api_base}/metrics`: {e}")
