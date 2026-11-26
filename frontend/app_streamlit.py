"""
Streamlit Web UI: Chẩn đoán bệnh trên lá cây nông nghiệp
Tầng hiển thị tương tác người dùng (Tier 3)
"""

import io
from PIL import Image
import requests
import streamlit as st

st.set_page_config(
    page_title="Chẩn đoán Bệnh Cây Trồng (ResNet-18)",
    page_icon="🌿",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"

st.title("🌿 Hệ Thống Chẩn Đoán Bệnh Trên Lá Cây Nông Nghiệp")
st.markdown("""
*Mô hình: **ResNet-18 (Long - Dataset v2 Leaf-Grouped)***
""")

# Kiểm tra trạng thái Backend
with st.sidebar:
    st.header("⚙️ Trạng thái Hệ thống")
    try:
        resp = requests.get(HEALTH_URL, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            st.success("🟢 Backend API: Sẵn sàng")
            st.info(f"📦 Mô hình: `{data.get('model_version')}`")
            st.caption(f"⚡ Thiết bị: `{data.get('device', 'cpu')}`")
        else:
            st.warning("🟡 Backend API: Đang bảo trì")
    except Exception:
        st.error("🔴 Không kết nối được Backend API (Port 8000). Hãy chạy `uvicorn backend.main:app --port 8000` trước.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Tải ảnh lá cây cần kiểm tra")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh lá cây đầu vào", use_container_width=True)

with col2:
    st.subheader("2. Kết quả chẩn đoán")
    if uploaded_file is not None:
        if st.button("🔍 Tiến hành Chẩn đoán", type="primary"):
            with st.spinner("Đang xử lý hình ảnh và phân tích..."):
                try:
                    img_bytes = uploaded_file.getvalue()
                    files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                    response = requests.post(API_URL, files=files, timeout=10)

                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"**Kết quả:** {res['predicted_class']}")
                        st.metric("Độ tin cậy (Confidence)", f"{res['confidence']*100:.2f}%")
                        st.caption(f"⏱️ Thời gian phản hồi (Latency): {res.get('inference_time_ms', 0):.2f} ms")

                        st.markdown("---")
                        st.markdown("**Top 3 khả năng cao nhất:**")
                        for item in res.get("top_3", []):
                            conf = float(item["confidence"])
                            st.progress(min(max(conf, 0.0), 1.0))
                            st.write(f"- **{item['class_name']}**: `{conf*100:.2f}%`")
                    else:
                        st.error(f"Lỗi phân tích từ server: {response.text}")
                except Exception as e:
                    st.error(f"Không thể kết nối đến API: {e}")
    else:
        st.info("Vui lòng tải lên ảnh lá cây ở cột bên trái để bắt đầu.")
