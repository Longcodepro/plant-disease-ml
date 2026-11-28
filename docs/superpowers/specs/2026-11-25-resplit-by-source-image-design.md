# Thiết kế: Chia lại dữ liệu theo ảnh gốc (chống rò rỉ train ↔ valid)

*Ngày: 2026-11-25 · Trạng thái: đã duyệt trong hội thoại · **Cập nhật 2026-11-26: chuyển sang cách B***

> **Cập nhật 2026-11-26 — đã làm theo cách B (bộ PlantVillage gốc), không phải cách A bên dưới.**
>
> - Nguồn: `github.com/spMohanty/PlantVillage-Dataset`, chỉ thư mục `raw/color` (38 lớp, 54.305 ảnh, không có
>   ảnh nhân bản), tải bằng sparse checkout vào `data/plantvillage_raw/`. Bản trên Kaggle
>   `abdallahalidev/plantvillage-dataset` là cùng dữ liệu (chỉ dùng `color`, không trộn `grayscale`/`segmented`).
> - Bộ trên Drive = Kaggle `vipoooool/new-plant-diseases-dataset` (Augmented) — **bị rò rỉ, không dùng nữa**.
> - Script: `scripts/split_plantvillage.py` (test: `python3 -m unittest tests/test_split_plantvillage.py`).
>   Gom nhóm "cùng một chiếc lá" theo **`leaf-map.json` chính thức** của PlantVillage (mã chụp → mã lá; một lá
>   tới 33 tấm khác mã), theo mã chụp sau `___` (bỏ ` Day N`) và theo ảnh trùng byte → băm khóa nhóm → 70/15/15.
>   Kết quả: `data_v2/` (gitignored) — train 38.039 / valid 8.147 / test 8.098; bỏ 21 ảnh trùng byte;
>   0 lá (kể cả theo leaf-map) nằm ở 2 phần; thứ tự lớp khớp `models/class_indices.json`.
>   Dấu vân tay cách chia: `4fc4a896d43ddbfa…`. Gói chia sẻ: `data/plantvillage_v2_split.tar.gz`
>   (sha256 `5c482ed9…`). Bản chia đầu tiên (vân tay `60bd66db…`, chưa dùng leaf-map) bị lọt 6.424 lá → đã xóa.
> - Giấy phép bộ gốc: CC BY-SA 3.0 → bản phát hành lại phải giữ CC BY-SA và trích dẫn Mohanty et al. 2016.
> - Phần "Thiết kế" bên dưới (bỏ đuôi `_flipLR`…) chỉ còn giá trị tham khảo cho cách A.

## Vì sao cần làm

Bộ dữ liệu của nhóm (`data.tar.gz` trên Google Drive, 70.295 train / 17.572 valid) là bản
*New Plant Diseases Dataset (Augmented)*: ảnh gốc PlantVillage đã bị **nhân bản (xoay, lật, đổi màu…)
TRƯỚC khi chia 80/20**. Hệ quả: bản sao của cùng một ảnh gốc nằm ở cả train lẫn valid
("lọt đề"), điểm valid bị thổi phồng (ConvNeXt-Tiny 99.95%) và không còn phân biệt được model nào tốt hơn.

Bằng chứng: tên file giữ mã ảnh gốc + đuôi phép biến đổi, ví dụ

```
train  Tomato___Late_blight/0a39aa48-…___RS_Late.B 5103.JPG
valid  Tomato___Late_blight/0a39aa48-…___RS_Late.B 5103_flipLR.JPG
```

Trong mẫu ~4% đầu của file nén đã có 59 ảnh gốc nằm ở cả hai phía. `data_audit.rs` không phát hiện
được vì nó so SHA-256 trên byte của file (bản lật có byte khác hẳn).

## Mục tiêu

- Tạo `data_v2/{train,valid,test}` trong đó **mọi bản sao của một ảnh gốc nằm cùng một phía**.
- Đúng thứ tự: **gom theo ảnh gốc → chia → tăng cường chỉ trên train** (bước tăng cường đã có sẵn,
  làm online trong `notebooks/data_pipeline.py`).
- Có tập **test thật** (~15%) để chấm một lần cuối, thay cho 33 ảnh hiện tại (mỗi ảnh = 3%).
- Ai chạy, ở máy nào (kể cả Colab), cũng ra **cùng một cách chia**.

## Không làm (đợt này)

- Không sửa/xóa `data/` hiện có (nhóm đang dùng).
- Không sửa `backend/rust/src/bin/data_audit.rs` (code của Khôi).
- Không sửa README / AI_COLLABORATION_GUIDE (tài liệu chung, cần nhóm đồng ý).
- Không train model.

## Điều tuyệt đối tránh (anti-goals)

- Một ảnh gốc xuất hiện ở hơn một phần (train/valid/test).
- Thứ tự lớp trong `data_v2` khác `models/class_indices.json` (lệch nhãn khi train/serve).
- Lỡ commit ~1 GB ảnh vào git.

## Ràng buộc

- Python hệ thống 3.14, chưa có numpy/torch → script **chỉ dùng thư viện chuẩn** (chạy được cả trên Colab).
- Pipeline chung (`build_dataloaders`) đọc `ImageFolder` từ `<data_dir>/train` và `<data_dir>/valid`
  → `data_v2` phải giữ đúng cấu trúc thư mục lớp, đủ 38 lớp ở mỗi phần.
- Script tải dữ liệu của nhóm xóa rồi ghi đè `data/test` (file đang được git theo dõi)
  → khi tải về máy này, chỉ giải nén `data/train` và `data/valid`.

## Các cách đã cân nhắc

| Cách | Ưu | Nhược | Kết luận |
|---|---|---|---|
| **A. Chia lại bộ của nhóm, gom bản sao theo mã ảnh gốc trong tên file** | Cùng nguồn với nhóm; gom chính xác ở mức ảnh; không cần tải thêm | Các lớp lớn đã bị tác giả Kaggle cắt bớt ảnh gốc | **Chọn** |
| B. Tải PlantVillage gốc (~54.300 ảnh) rồi chia từ đầu | Sạch, đủ ảnh gốc | Thêm nguồn tải khác, cả nhóm phải đổi bộ dữ liệu; không cần thiết vì tên file đã đủ thông tin | Bỏ |
| C. Gom bản sao bằng so nội dung ảnh (perceptual hash) | Dùng được khi tên file không có thông tin | Gần đúng, nặng hơn, cần thư viện ngoài | Để sau |

## Thiết kế

1. **Lấy mã ảnh gốc:** bỏ đuôi biến đổi khỏi tên file (không tính phần mở rộng). Chỉ bỏ các đuôi
   thuộc danh sách đã biết: `_<số>deg`, `_flipLR`, `_flipTB`, `_new…` (vd. `_new30degFlipLR`,
   `_newPixel25`, `_newGRR`). Đuôi lạ thì **giữ nguyên** (an toàn: có thể gom thiếu, không bao giờ gom nhầm)
   và được liệt kê trong báo cáo để người xem lại.
   Khóa nhóm = `<tên lớp>/<mã ảnh gốc>`.
2. **Mỗi ảnh gốc giữ 1 tấm:** ưu tiên tấm không có đuôi (ảnh gốc); nếu nhóm không có ảnh gốc thì giữ
   tấm đứng đầu theo thứ tự tên. Các bản sao còn lại bỏ đi (train sẽ được xoay/lật online).
3. **Chia bằng cách băm khóa nhóm** (ý tưởng của Géron, Chương 2: băm mã định danh để tập test ổn định):
   lấy SHA-256 của khóa nhóm → đổi 8 byte đầu thành số trong [0, 1) → `< 0.70` train,
   `< 0.85` valid, còn lại test. Cùng khóa ⇒ cùng phần ⇒ bản sao không thể tách đôi; không phụ thuộc
   thứ tự file hay máy chạy. (Không dùng `hash()` của Python vì nó đổi giá trị mỗi lần chạy.)
4. **Tự kiểm tra trước khi ghi:**
   - Mỗi khóa nhóm chỉ nằm ở 1 phần.
   - Kiểm tra chéo độc lập: mã UUID 36 ký tự đầu tên file (nếu có) không xuất hiện ở 2 phần.
   - Cả 3 phần có đủ và đúng danh sách lớp của nguồn.
   Sai bất kỳ điều nào ⇒ dừng, không ghi gì.
5. **Ghi kết quả:** `data_v2/<phần>/<lớp>/<tên file>` bằng hardlink (không tốn thêm ổ đĩa; tự chuyển
   sang copy nếu hardlink không được). Kèm `data_v2/manifest.csv` (ảnh nào vào đâu) và
   `data_v2/split_report.json` (số ảnh từng lớp, thống kê đuôi, đuôi lạ, **dấu vân tay của cách chia**
   để cả nhóm so xem có chia giống nhau không). Từ chối chạy nếu thư mục đích đã có dữ liệu.
6. **Cách dùng:** `python scripts/resplit_by_source.py --src data --dst data_v2 [--dry-run]`,
   rồi trong notebook đổi `build_dataloaders(data_dir="data_v2")`.

## Hạn chế đã biết

- PlantVillage đôi khi chụp **cùng một chiếc lá nhiều lần** với mã khác nhau; tên file không cho biết
  nên chưa loại được (cần cách C).
- Điểm trên `data_v2` **không so được** với điểm cũ; mọi model (kể cả ConvNeXt) phải train lại trên `data_v2/train`.

## Để người làm tự quyết

- Tên hàm, cách in bảng báo cáo, chi tiết test.
