"""
Streamlit Web UI: PlantGuard AI — Chẩn đoán Bệnh Lá Cây Nông Nghiệp (Tier 3)
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
    page_title="PlantGuard AI — Chẩn đoán Bệnh Lá Cây",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT_DIR / "data" / "test"
API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# ==============================================================================
# MODERN AGRITECH CSS THEME (Botanical Emerald + Elevated Cards + Custom Bars)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Ẩn sidebar & chrome thừa của Streamlit */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; height: 2rem; }

    /* Nền trang nhẹ nhàng, sạch sẽ */
    .stApp {
        background: linear-gradient(180deg, #f3f7f4 0%, #f8faf9 45%, #ffffff 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #047857 52%, #0f766e 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 1.15rem;
        box-shadow: 0 12px 30px rgba(6, 78, 59, 0.16);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .hero-sub {
        color: rgba(236, 253, 245, 0.88);
        font-size: 0.96rem;
        margin: 0.45rem 0 1rem 0;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(6px);
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #ecfdf5;
    }
    .status-dot {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(16, 185, 129, 0.22);
        border: 1px solid rgba(110, 231, 183, 0.45);
        color: #d1fae5;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-dot.offline {
        background: rgba(239, 68, 68, 0.22);
        border-color: rgba(252, 165, 165, 0.45);
        color: #fee2e2;
    }

    /* Tabs hiện đại thay gạch chân đỏ mặc định */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #eaf1ec;
        padding: 0.35rem;
        border-radius: 0.85rem;
        border: none;
        margin-bottom: 1.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.65rem;
        padding: 0.55rem 1.15rem;
        font-weight: 600;
        color: #1f3b2c;
        border: none !important;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #047857 !important;
        box-shadow: 0 2px 8px rgba(6, 78, 59, 0.10);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Card containers */
    .panel-card {
        background: #ffffff;
        border: 1px solid #e2ece6;
        border-radius: 1rem;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    .panel-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f291e;
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* Diagnosis Result Card */
    .diag-card {
        padding: 1.25rem 1.4rem;
        border-radius: 1rem;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #059669;
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        margin-bottom: 1.1rem;
    }
    .diag-card.disease {
        border-color: #fed7aa;
        border-left-color: #ea580c;
        background: linear-gradient(135deg, #fff7ed 0%, #fef2f2 100%);
    }
    .diag-card.uncertain {
        border-color: #fde68a;
        border-left-color: #d97706;
        background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
    }
    .diag-plant {
        font-size: 0.86rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
    }
    .diag-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.2rem 0 0.7rem 0;
        line-height: 1.25;
    }
    .diag-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(15, 23, 42, 0.08);
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #1e293b;
    }
    .meta-chip.cache {
        background: #dbeafe;
        border-color: #93c5fd;
        color: #1d4ed8;
    }

    /* Custom Top-3 Probability Bars */
    .prob-row { margin-bottom: 0.75rem; }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.28rem;
    }
    .prob-track {
        width: 100%;
        height: 10px;
        background: #e5ece8;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
    }
    .prob-fill.alt {
        background: linear-gradient(90deg, #64748b 0%, #94a3b8 100%);
    }

    /* Heatmap legend */
    .heat-legend {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #475569;
    }
    .heat-bar {
        flex: 1;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, #1d4ed8 0%, #06b6d4 30%, #22c55e 55%, #facc15 78%, #dc2626 100%);
    }

    /* KPI Stat Cards (Monitoring & Batch) */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2ece6;
        border-top: 4px solid #059669;
        border-radius: 0.9rem;
        padding: 1rem 1.15rem;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.2rem 0;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #059669;
        font-weight: 600;
    }

    /* Nút Primary màu xanh ngọc nổi bật */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border: none !important;
        border-radius: 0.7rem !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.28) !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    /* Bo góc ảnh */
    [data-testid="stImage"] img {
        border-radius: 0.85rem;
        border: 1px solid #e2ece6;
    }

    /* Empty state */
    .empty-state {
        border: 2px dashed #cbd5e1;
        border-radius: 1rem;
        padding: 3rem 1.5rem;
        text-align: center;
        color: #64748b;
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_html(html: str) -> None:
    """Nén khoảng trắng đầu dòng trước khi render để Markdown không bao giờ hiểu nhầm thành code block."""
    compact = "".join(line.strip() for line in html.splitlines())
    st.markdown(compact, unsafe_allow_html=True)


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


def fetch_health() -> Tuple[bool, str]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=1.5)
        if r.status_code == 200:
            data = r.json()
            return bool(data.get("model_loaded")), str(data.get("model_version", "ONNX"))
    except Exception:
        pass
    return False, "Offline"


# ── Hero Banner (tự động kiểm tra trạng thái ngầm, không hiện sidebar thừa) ──
is_online, model_ver = fetch_health()
status_html = (
    f'<span class="status-dot">● Sẵn sàng · {model_ver}</span>'
    if is_online
    else '<span class="status-dot offline">● Chưa kết nối máy chủ suy luận</span>'
)

render_html(
    f"""
    <div class="hero-banner">
        <div class="hero-top">
            <h1 class="hero-title">🌿 PlantGuard AI — Chẩn Đoán Bệnh Lá Cây</h1>
            {status_html}
        </div>
        <p class="hero-sub">
            Hệ thống Thị giác Máy tính nhận diện sớm 38 loại cây trồng &amp; bệnh lý từ ảnh chụp lá cây,
            tích hợp bản đồ nhiệt giải thích XAI (CAM / Grad-CAM) và giám sát độ trễ trực tuyến.
        </p>
        <div class="hero-pills">
            <span class="hero-pill">🧠 ConvNeXt-Tiny · ONNX INT8 (32.3 MB)</span>
            <span class="hero-pill">🎯 99.95% Val Macro-F1 · 33/33 Test</span>
            <span class="hero-pill">⚡ 23.8 ms / ảnh trên Apple M2</span>
            <span class="hero-pill">🔥 XAI CAM / Grad-CAM (~26 ms)</span>
        </div>
    </div>
    """
)

if not is_online:
    st.warning("Máy chủ chẩn đoán chưa khởi động. Hãy chạy `uvicorn backend.main:app --port 8000` rồi tải lại trang.")

tab_single, tab_batch, tab_monitor = st.tabs(
    ["🔍 Chẩn đoán & Giải thích XAI", "📦 Chẩn đoán hàng loạt", "📊 Giám sát hệ thống"]
)

# ==============================================================================
# TAB 1: CHẨN ĐOÁN 1 ẢNH + HEATMAP XAI
# ==============================================================================
with tab_single:
    col_in, col_out = st.columns([1, 1.25], gap="large")

    img_bytes: bytes | None = None
    img_name = "leaf.jpg"
    img_mime = "image/jpeg"

    with col_in:
        st.markdown('<div class="panel-heading">📷 1. Chọn ảnh lá cây cần kiểm tra</div>', unsafe_allow_html=True)
        samples = load_sample_files()
        source_mode = (
            st.radio("Nguồn ảnh", ["Ảnh mẫu có sẵn", "Tải ảnh từ máy"], horizontal=True, label_visibility="collapsed")
            if samples
            else "Tải ảnh từ máy"
        )

        if source_mode == "Tải ảnh từ máy":
            uploaded = st.file_uploader("Kéo thả hoặc chọn ảnh JPG, PNG", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                img_bytes = uploaded.getvalue()
                img_name = uploaded.name
                img_mime = uploaded.type or "image/jpeg"
        else:
            chosen_name = st.selectbox("Chọn nhanh ảnh kiểm thử độc lập (data/test)", [p.name for p in samples])
            chosen_path = SAMPLE_DIR / chosen_name
            img_bytes = chosen_path.read_bytes()
            img_name = chosen_path.name
            img_mime = "image/png" if chosen_path.suffix.lower() == ".png" else "image/jpeg"

        if img_bytes is not None:
            st.image(Image.open(io.BytesIO(img_bytes)), caption=f"Ảnh đầu vào · {img_name}", use_container_width=True)

    with col_out:
        st.markdown('<div class="panel-heading">🩺 2. Kết quả phân tích AI</div>', unsafe_allow_html=True)
        if img_bytes is not None:
            ctrl1, ctrl2 = st.columns([1.2, 1])
            with ctrl1:
                show_xai = st.toggle("🔥 Hiển thị Heatmap XAI", value=True)
            with ctrl2:
                run_btn = st.button("🔍 Chẩn đoán ngay", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Đang phân tích mẫu lá..."):
                    try:
                        files = {"file": (img_name, img_bytes, img_mime)}
                        pred_resp = requests.post(f"{API_BASE}/predict", files=files, timeout=15)
                        if pred_resp.status_code != 200:
                            st.error(f"Lỗi phân tích ({pred_resp.status_code}): {pred_resp.text}")
                        else:
                            res = pred_resp.json()
                            plant, condition, is_healthy = format_label(res["predicted_class"])
                            is_uncertain = bool(res.get("is_uncertain", False))
                            card_cls = "uncertain" if is_uncertain else ("" if is_healthy else "disease")
                            icon = "⚠️" if is_uncertain else ("✅" if is_healthy else "🩺")
                            cache_badge = '<span class="meta-chip cache">⚡ Cache Hit</span>' if res.get("cache_hit") else ""

                            render_html(
                                f"""
                                <div class="diag-card {card_cls}">
                                    <div class="diag-plant">🌱 {plant}</div>
                                    <div class="diag-title">{icon} {condition}</div>
                                    <div class="diag-meta">
                                        <span class="meta-chip">🎯 Độ tin cậy: {res['confidence']*100:.2f}%</span>
                                        <span class="meta-chip">⏱️ {res['inference_time_ms']} ms</span>
                                        {cache_badge}
                                    </div>
                                </div>
                                """
                            )

                            if res.get("warning"):
                                st.warning(res["warning"])

                            # Custom Top-3 Probability Bars
                            bars_html = ['<div class="panel-card"><div class="panel-heading">📊 Top 3 chẩn đoán xác suất cao nhất</div>']
                            for idx, item in enumerate(res.get("top_3", [])):
                                p_name, c_name, _ = format_label(item["class_name"])
                                pct = min(max(float(item["confidence"]) * 100.0, 0.0), 100.0)
                                fill_cls = "" if idx == 0 else "alt"
                                bars_html.append(
                                    f"""
                                    <div class="prob-row">
                                        <div class="prob-header">
                                            <span>{idx+1}. {p_name} — {c_name}</span>
                                            <span>{pct:.2f}%</span>
                                        </div>
                                        <div class="prob-track">
                                            <div class="prob-fill {fill_cls}" style="width:{pct:.1f}%"></div>
                                        </div>
                                    </div>
                                    """
                                )
                            bars_html.append("</div>")
                            render_html("".join(bars_html))

                            if show_xai:
                                with st.spinner("Đang tạo bản đồ nhiệt giải thích vùng bệnh..."):
                                    ex_resp = requests.post(
                                        f"{API_BASE}/explain",
                                        files={"file": (img_name, img_bytes, img_mime)},
                                        timeout=30,
                                    )
                                if ex_resp.status_code == 200:
                                    ex = ex_resp.json()
                                    heat_img = Image.open(io.BytesIO(base64.b64decode(ex["heatmap_base64"])))
                                    render_html(
                                        f'<div class="panel-heading">🔥 Giải thích trực quan XAI '
                                        f'<span style="font-size:0.8rem;font-weight:500;color:#64748b;">({ex["method"]} · {ex["explain_time_ms"]} ms)</span></div>'
                                    )
                                    c1, c2 = st.columns(2, gap="medium")
                                    c1.image(Image.open(io.BytesIO(img_bytes)), caption="Ảnh gốc", use_container_width=True)
                                    c2.image(heat_img, caption="Vùng mô hình tập trung", use_container_width=True)
                                    render_html(
                                        '<div class="heat-legend"><span>Ít ảnh hưởng</span><div class="heat-bar"></div><span>Vùng vết bệnh quyết định</span></div>'
                                    )
                                else:
                                    st.info("Backend hiện tại chưa bật endpoint `/explain`.")
                    except Exception as e:
                        st.error(f"Không thể kết nối máy chủ suy luận: {e}")
            else:
                render_html(
                    """
                    <div class="empty-state">
                        <div style="font-size:2.2rem;margin-bottom:0.4rem;">🌿</div>
                        <div style="font-weight:700;color:#1e293b;margin-bottom:0.25rem;">Ảnh đã sẵn sàng để phân tích</div>
                        <div style="font-size:0.9rem;">Nhấn nút <b>🔍 Chẩn đoán ngay</b> ở phía trên để nhận kết quả &amp; bản đồ nhiệt XAI.</div>
                    </div>
                    """
                )
        else:
            render_html(
                """
                <div class="empty-state">
                    <div style="font-size:2.2rem;margin-bottom:0.4rem;">📷</div>
                    <div style="font-weight:700;color:#1e293b;margin-bottom:0.25rem;">Chưa có ảnh đầu vào</div>
                    <div style="font-size:0.9rem;">Hãy chọn ảnh mẫu hoặc tải lên ảnh lá cây ở cột bên trái.</div>
                </div>
                """
            )


# ==============================================================================
# TAB 2: CHẨN ĐOÁN HÀNG LOẠT (/predict/batch)
# ==============================================================================
with tab_batch:
    st.markdown('<div class="panel-heading">📦 Suy luận nhiều ảnh trong 1 batch ONNX duy nhất</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Chọn tối đa 32 ảnh lá cây (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        if st.button(f"🚀 Chẩn đoán ngay {min(len(batch_files), 32)} ảnh", type="primary"):
            with st.spinner(f"Đang xử lý batch {min(len(batch_files), 32)} ảnh..."):
                try:
                    selected = batch_files[:32]
                    payload = [("files", (f.name, f.getvalue(), f.type or "image/jpeg")) for f in selected]
                    resp = requests.post(f"{API_BASE}/predict/batch", files=payload, timeout=60)
                    if resp.status_code != 200:
                        st.error(f"Lỗi batch API ({resp.status_code}): {resp.text}")
                    else:
                        data = resp.json()
                        avg_ms = data["total_inference_time_ms"] / max(1, data["count"])
                        k1, k2, k3 = st.columns(3)
                        k1.markdown(
                            f'<div class="kpi-card"><div class="kpi-label">Số ảnh xử lý</div>'
                            f'<div class="kpi-val">{data["count"]}</div><div class="kpi-sub">1 lần gọi ONNX batch</div></div>',
                            unsafe_allow_html=True,
                        )
                        k2.markdown(
                            f'<div class="kpi-card"><div class="kpi-label">Tổng thời gian</div>'
                            f'<div class="kpi-val">{data["total_inference_time_ms"]} ms</div><div class="kpi-sub">End-to-end model</div></div>',
                            unsafe_allow_html=True,
                        )
                        k3.markdown(
                            f'<div class="kpi-card"><div class="kpi-label">Trung bình / ảnh</div>'
                            f'<div class="kpi-val">{avg_ms:.1f} ms</div><div class="kpi-sub">~{1000/max(avg_ms,0.1):.1f} FPS</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

                        rows = []
                        cols = st.columns(4)
                        for idx, (f, item) in enumerate(zip(selected, data["results"])):
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

                        csv_buf = io.StringIO()
                        writer = csv.DictWriter(csv_buf, fieldnames=["file", "plant", "condition", "confidence", "uncertain"])
                        writer.writeheader()
                        writer.writerows(rows)
                        st.download_button(
                            "⬇️ Tải bảng kết quả CSV",
                            data=csv_buf.getvalue().encode("utf-8-sig"),
                            file_name="plant_disease_batch_results.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Lỗi kết nối `/predict/batch`: {e}")
    else:
        render_html(
            """
            <div class="empty-state">
                <div style="font-size:2rem;margin-bottom:0.4rem;">📂</div>
                <div style="font-weight:700;color:#1e293b;">Kéo thả nhiều ảnh lá cây vào khung trên</div>
                <div style="font-size:0.9rem;">Hệ thống sẽ chạy gộp toàn bộ ảnh trong 1 batch ONNX và xuất bảng CSV kết quả.</div>
            </div>
            """
        )


# ==============================================================================
# TAB 3: LIVE MONITORING (/metrics)
# ==============================================================================
with tab_monitor:
    h_left, h_right = st.columns([5, 1])
    h_left.markdown('<div class="panel-heading">📊 Giám sát hiệu năng &amp; tín hiệu Data Drift trực tuyến</div>', unsafe_allow_html=True)
    h_right.button("🔄 Làm mới", use_container_width=True)

    try:
        m_resp = requests.get(f"{API_BASE}/metrics", timeout=5)
        if m_resp.status_code == 200:
            m = m_resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Tổng Requests</div>'
                f'<div class="kpi-val">{m["total_requests"]}</div>'
                f'<div class="kpi-sub">{m["total_images_predicted"]} ảnh đã xử lý</div></div>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Latency p50</div>'
                f'<div class="kpi-val">{m["latency_ms"]["p50"]} ms</div>'
                f'<div class="kpi-sub">p95: {m["latency_ms"]["p95"]} ms · avg: {m["latency_ms"]["avg"]} ms</div></div>',
                unsafe_allow_html=True,
            )
            c3.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Cache Hit Rate</div>'
                f'<div class="kpi-val">{m["cache_hit_rate"]*100:.1f}%</div>'
                f'<div class="kpi-sub">⚡ {m["cache_hits"]} lần trả tức thì</div></div>',
                unsafe_allow_html=True,
            )
            c4.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Uncertain / Drift</div>'
                f'<div class="kpi-val">{m["uncertain_rate"]*100:.1f}%</div>'
                f'<div class="kpi-sub">{m["uncertain_count"]} ảnh dưới ngưỡng</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.caption(
                f"⏱️ Uptime: `{m['uptime_seconds']}s` · Runtime: `{m['backend_type'].upper()}` · Model: `{m['model_version']}`"
            )

            top_cls = m.get("top_predicted_classes") or {}
            if top_cls:
                st.markdown('<div class="panel-heading" style="margin-top:0.5rem;">🌿 Top lớp bệnh được chẩn đoán nhiều nhất</div>', unsafe_allow_html=True)
                pretty_counts = {
                    f"{format_label(k)[0]} — {format_label(k)[1]}": v for k, v in top_cls.items()
                }
                st.bar_chart(pretty_counts, color="#059669")
            else:
                render_html(
                    """
                    <div class="empty-state">
                        <div style="font-size:2rem;margin-bottom:0.4rem;">📈</div>
                        <div style="font-weight:700;color:#1e293b;">Chưa có dữ liệu thống kê</div>
                        <div style="font-size:0.9rem;">Hãy chẩn đoán thử vài ảnh ở Tab 1 rồi bấm <b>🔄 Làm mới</b> để xem biểu đồ trực tiếp.</div>
                    </div>
                    """
                )
        else:
            st.warning(f"Máy chủ không trả về `/metrics` (HTTP {m_resp.status_code}).")
    except Exception as e:
        st.error(f"Không thể lấy dữ liệu giám sát: {e}")
