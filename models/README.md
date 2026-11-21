# Models Directory

Thư mục lưu trữ các Model Artifacts, Trọng số (Checkpoints) và Metadata:
- `class_indices.json`: Ánh xạ 38 nhãn giữa index số nguyên và tên lớp bệnh / cây trồng.
- `baseline1_simple_cnn.pth`: Checkpoint của Baseline 1 (Vanilla CNN từ đầu, `1.6 MB`).
- `convnext_tiny_khoi_int8.onnx`: **Mô hình chủ lực `ConvNeXt-Tiny` đã xuất ONNX & nén INT8** (`32.3 MB`, đạt `33/33 = 100%` trên `data/test`, tốc độ `23.81 ms/ảnh` ≈ `42 FPS` trên Apple M2 CPU — sẵn sàng cắm chạy ngay FastAPI hoặc Rust API).
- `convnext_tiny_khoi.pth` / `convnext_tiny_khoi.onnx` (`106.3 MB`, Val Macro-F1 `99.95%`): Vượt giới hạn `100 MB` của GitHub nên được ignore trong `.gitignore` và lưu trên Google Drive của nhóm.
