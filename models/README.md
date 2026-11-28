# Thư Mục Mô Hình (Models Directory) - Nhánh feat/resnet18-long

Tác giả: **Nguyễn Thành Long**  
Kiến trúc nòng cốt: **ResNet-18 v2 & YOLOv8n Leaf Detector**

---

## 1. Danh Mục Trọng Số & Tệp Mô Hình (Model Artifacts)

- `class_indices.json`: Bảng ánh xạ 38 nhãn giữa chỉ số số nguyên và tên lớp bệnh / cây trồng chuẩn.
- `model_registry.csv`: Sổ đăng ký chính thức phiên bản và các chỉ số kiểm thử.
- `finetune_history_resnet18_v2.csv`: Lịch sử chi tiết 12 epochs quá trình thích nghi miền dữ liệu 2:1.
- `epoch_history.csv`: Lịch sử huấn luyện qua các epoch của các mô hình cơ sở.

### Mô hình Phân loại Bệnh học (ResNet-18):
- `resnet18_long_v2_finetuned_best.pth` (**44.8 MB**): Checkpoint PyTorch tối ưu nhất tại **Epoch 8** (Val Macro F1: 48.56%), lưu giữ toàn bộ trọng số, optimizer và scheduler.
- `resnet18_long_v2_finetuned_best_int8.onnx` (**10.7 MB**): Mô hình ONNX lượng tử hóa động INT8, giảm kích thước $4\times$, tối ưu cho thiết bị biên (Edge/CPU).
- `resnet18_long_v2_run_01.pth` (**44.8 MB**): Mô hình ResNet-18 v2 chuẩn phòng lab (đạt 99.16% F1 trên PlantVillage).
- `baseline1_simple_cnn.pth` (**1.7 MB**): Mô hình mạng tích chập cơ sở (Vanilla CNN) làm mốc đối chứng tối thiểu.

### Mô hình Định vị Phiến lá (YOLOv8n):
- `yolo/yolov8n_leaf.pt` (**6.0 MB**): Mô hình YOLOv8n định vị phiến lá ngoại cảnh ($mAP_{50} = 85.3\%$), làm bộ lọc không gian trước khi đưa vào ResNet-18.
