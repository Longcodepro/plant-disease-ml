# TÀI LIỆU NGỮ CẢNH & QUY TẮC DỰ ÁN (PROJECT CONTEXT & HANDOVER)
*Dành cho AI Assistant / Antigravity Desktop tiếp nhận dự án*

---

## 1. THÔNG TIN DỰ ÁN VÀ MỤC TIÊU
- **Tên dự án chính:** Chẩn đoán bệnh trên lá cây nông nghiệp (Plant Disease Diagnosis).
- **Tập dữ liệu:** PlantVillage Dataset (38 lớp cây trồng & bệnh lý, ~87,000 ảnh).
- **Kiến trúc hệ thống (3-Tier Decoupled Architecture):**
  - **Tầng 1 (Notebooks/Training):** Nghiên cứu, huấn luyện và đánh giá mô hình (`notebooks/`).
  - **Tầng 2 (Backend REST API):** FastAPI phục vụ suy luận độ trễ thấp (`backend/`).
  - **Tầng 3 (Frontend Web UI):** Streamlit giao diện trực quan cho người dùng (`frontend/`).
- **Mục tiêu học tập song song:** Học lại toàn bộ Machine Learning từ con số 0 theo phương pháp **bản chất, trực quan, có thể kéo thả tham số**. Nền tảng này dùng chung cho cả mảng **Thị giác máy tính (Computer Vision)** trong dự án này và mảng **Xử lý ngôn ngữ tự nhiên (NLP)** trong môn học tiếp theo.

---

## 2. TÀI LIỆU THAM KHẢO CỐT LÕI (CÓ SẴN TRÊN MÁY)
Mọi tư duy thiết kế hệ thống và mô hình đều phải bám sát 2 cuốn giáo trình sau:
1. **Designing Machine Learning Systems (Chip Huyen):**
   - Đã được bóc tách thành các file Markdown từng chương tại:
     `/home/long/Documents/University/BOOK/Designing Machine Learning Systems/Chapter(markdown)`
2. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron):**
   - Tài liệu tham khảo về triển khai kỹ thuật, pipeline và mô hình học sâu.

---

## 3. QUY TẮC SƯ PHẠM & HƯỚNG DẪN HỌC TẬP (BẮT BUỘC TUÂN THỦ)
1. **Chia nhỏ vấn đề, giải thích từng chút một:** Tuyệt đối không "xổ" một lượng kiến thức quá lớn hoặc quá nhiều câu hỏi trong một lần trả lời. Luôn tương tác 2 chiều, đợi người học hiểu rồi mới sang ý tiếp theo.
2. **Ngôn ngữ bình dân, tránh thuật ngữ hàn lâm:** Coi người học như người mới bắt đầu từ con số 0. Dùng ví dụ đời thường để giải thích bản chất trước khi gắn nhãn khái niệm toán học.
3. **Quy ước thư mục học tập (`hoc_tap_ml/`):**
   - Mỗi phần học là một thư mục con đặt tên bằng **Tiếng Anh** (Ví dụ: `01_what_is_machine_learning`).
4. **Trực quan hóa sạch sẽ & tương tác:**
   - Ưu tiên vẽ đồ thị trực quan, sinh động.
   - **Tuyệt đối không chèn các dòng giải thích nhỏ nhặt, rườm rà lên hình vẽ** gây rối mắt.
   - Luôn thiết kế các thanh trượt/tham số kéo thả (Sliders) để người học tự tay kéo và quan sát sự biến đổi.
5. **Lệnh Linux/Terminal:** Người học dùng Linux Mint và đang làm quen lệnh. Khi đưa câu lệnh terminal, phải giải thích rõ từng phần và cờ lệnh.

---

## 4. HAI BỘ VÍ DỤ CỐ ĐỊNH XUYÊN SUỐT (CANONICAL BENCHMARKS)
Để tránh bị rối khi đổi bài toán liên tục, toàn bộ phần học nền tảng sẽ dùng cố định 2 ví dụ:
1. **Ví dụ Hồi quy (Dự đoán số liên tục):**
   - Bài toán: Dự đoán **Tiền điện ($y$)** theo **Số giờ bật điều hòa ($x$)**.
   - Mục đích: Học về công thức đường thẳng ($y = wx + b$), Hàm mất mát (Loss), Đạo hàm và Gradient Descent.
2. **Ví dụ Phân loại (Nhóm/Nhãn - Cốt lõi của Dự án Lá cây & NLP):**
   - Bài toán: Phân loại **Lá cây Khỏe mạnh ($0$)** vs **Lá cây Bị bệnh ($1$)** dựa trên 2 đặc trưng: độ xanh của lá ($x_1$) và diện tích đốm nâu ($x_2$).
   - Mục đích: Học về Ranh giới phân loại (Decision Boundary), Confusion Matrix (TP, FP, FN, TN), Accuracy Paradox, Precision, Recall, F1-Score, và Hiện tượng Học vẹt (Overfitting).

---

## 5. CẤU TRÚC THƯ MỤC DỰ ÁN HIỆN TẠI
```text
plant-disease-ml/
├── data/
│   ├── train/                      # 38 thư mục lớp (70,295 ảnh)
│   ├── valid/                      # 38 thư mục lớp (17,572 ảnh)
│   └── test/                       # 33 ảnh kiểm thử độc lập
├── notebooks/                      # Tầng 1: Code thử nghiệm mô hình chính
├── models/                         # Lưu trữ checkpoints (.pth) & metadata
│   ├── class_indices.json          # Ánh xạ 38 nhãn
│   └── baseline1_simple_cnn.pth    # Checkpoint Baseline 1
├── backend/                        # Tầng 2: FastAPI Prediction Service
│   ├── main.py, schemas.py, service.py
├── frontend/                       # Tầng 3: Streamlit Web UI
│   └── app.py
├── hoc_tap_ml/                     # Vùng học tập nền tảng ML trực quan
│   └── 01_what_is_machine_learning/
└── PROJECT_CONTEXT.md              # File ngữ cảnh này
```

---

## 6. TIẾN ĐỘ HIỆN TẠI & BƯỚC TIẾP THEO
- **Đã làm:** 
  - Khám phá dữ liệu (EDA), Data Pipeline chống rò rỉ dữ liệu (Chương 3, 4, 5).
  - Hoàn thiện khung 3-Tier (`backend` FastAPI, `frontend` Streamlit, `models`).
  - Đã xong phần mở đầu của **Bài 1 (`hoc_tap_ml/01_what_is_machine_learning`)**: Người học đã nắm bản chất việc máy học là tìm bộ số $w$ và $b$ để giảm thiểu sai số (Loss).
- **Bước tiếp theo cần học:** 
  - Làm thế nào máy tính tự động tìm ra $w$ và $b$ tối ưu mà không cần người kéo tay? (Giới thiệu tự nhiên về **Hàm Loss** và ý tưởng **Gradient Descent - quả bóng lăn xuống dốc** qua đồ thị trực quan).
