"""
Agronomy Knowledge Base & Label Formatter for PlantGuard AI.
Cung cấp hàm chuẩn hóa nhãn PlantVillage ("Plant___Condition") và khuyến nghị nông học bằng tiếng Việt.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_HEALTHY_TEMPLATE_TIPS: Dict[str, List[str]] = {
    "treatment": [
        "Cây đang khỏe mạnh, không cần phun thuốc bảo vệ thực vật.",
        "Tiếp tục duy trì chế độ tưới tiêu và bón phân cân đối N-P-K theo giai đoạn sinh trưởng.",
    ],
    "prevention": [
        "Kiểm tra định kỳ mặt dưới lá 1–2 lần/tuần để phát hiện sớm dấu hiệu sâu bệnh.",
        "Giữ vườn thông thoáng, tỉa cành già sát gốc và tránh tưới đẫm nước lên tán lá vào chiều tối.",
        "Vệ sinh dụng cụ cắt tỉa và luân canh hợp lý giữa các vụ trồng.",
    ],
}

# Khuyến nghị chi tiết theo từng bệnh cụ thể (bao phủ toàn bộ nhóm bệnh trong 38 lớp PlantVillage)
_DISEASE_ADVISORIES: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "severity": "medium",
        "summary": "Bệnh ghẻ táo do nấm Venturia inaequalis gây đốm nâu ô-liu trên lá và làm sần vỏ quả, phát triển mạnh khi thời tiết ẩm mát.",
        "treatment": [
            "Cắt bỏ và tiêu hủy ngay các lá, chồi non có vết bệnh nặng để giảm nguồn bào tử.",
            "Phun thuốc trừ nấm gốc Captan, Mancozeb hoặc hoạt chất Myclobutanil theo đúng liều khuyến cáo.",
            "Luân phiên hoạt chất giữa các lần phun cách nhau 7–10 ngày để tránh kháng thuốc.",
        ],
        "prevention": [
            "Thu gom và ủ hủy toàn bộ lá rụng cuối vụ nhằm cắt đứt nguồn nấm qua đông.",
            "Tỉa tán thông thoáng để ánh nắng xuyên vào trong tán lá và lá mau khô sau mưa.",
            "Ưu tiên trồng giống táo kháng bệnh ghẻ và tránh tưới phun mưa trực tiếp lên tán.",
        ],
    },
    "Apple___Black_rot": {
        "severity": "high",
        "summary": "Bệnh thối đen trên táo do nấm Botryosphaeria obtusa gây đốm mắt ếch trên lá, loét cành và thối khô trái.",
        "treatment": [
            "Cắt tỉa cành bị loét, thu gom quả thối khô trên cây và dưới đất đem tiêu hủy.",
            "Phun thuốc trừ nấm phổ rộng chứa Captan, Thiophanate-methyl hoặc Pyraclostrobin.",
            "Quét vôi hoặc dung dịch đồng lên vết cắt lớn trên thân cành để ngăn nấm xâm nhập.",
        ],
        "prevention": [
            "Loại bỏ cây dại, cành chết xung quanh vườn vốn là ký chủ phụ của nấm.",
            "Bón phân cân đối giúp cây khỏe, tăng sức chống chịu và hạn chế vết thương cơ giới.",
            "Vệ sinh kéo cắt cành bằng cồn 70% sau khi cắt tỉa cây bệnh.",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "severity": "medium",
        "summary": "Bệnh gỉ sắt táo do nấm Gymnosporangium juniperi-virginianae tạo đốm vàng cam sáng ở mặt trên lá và tua bào tử mặt dưới.",
        "treatment": [
            "Phun thuốc trừ nấm đặc trị gỉ sắt nhóm Triazole (Myclobutanil, Propiconazole) hoặc Mancozeb ngay khi thấy đốm cam đầu tiên.",
            "Ngắt bỏ các lá nhiễm nặng nếu diện tích nhỏ để giảm áp lực lây lan.",
        ],
        "prevention": [
            "Tránh trồng cây bách xù/tuyết tùng (ký chủ trung gian) trong bán kính gần vườn táo.",
            "Phun phòng sớm vào giai đoạn cây ra nụ hồng và sau khi rụng cánh hoa.",
            "Chọn các giống táo có khả năng kháng hoặc ít mẫn cảm với bệnh gỉ sắt.",
        ],
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "severity": "medium",
        "summary": "Bệnh phấn trắng trên anh đào do nấm Podosphaera clandestina phủ lớp bột trắng mặt lá, làm lá xoăn và chồi non kém phát triển.",
        "treatment": [
            "Tỉa bỏ chồi non bị phủ phấn trắng nặng và đem tiêu hủy xa khu vực trồng.",
            "Phun lưu huỳnh (Sulfur), Potassium bicarbonate hoặc thuốc gốc Triazole vào lúc trời mát.",
        ],
        "prevention": [
            "Không bón thừa đạm gây ra nhiều chồi non mềm dễ nhiễm nấm.",
            "Cắt tỉa tán giúp lưu thông không khí tốt, giảm ẩm độ trong tán lá.",
        ],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "severity": "high",
        "summary": "Bệnh đốm lá xám trên ngô do nấm Cercospora zeae-maydis gây vết cháy hình chữ nhật song song gân lá, làm giảm mạnh khả năng quang hợp.",
        "treatment": [
            "Phun thuốc trừ nấm nhóm Strobilurin (Azoxystrobin) hoặc Triazole khi vết bệnh xuất hiện ở lá dưới bắp trước trỗ cờ.",
            "Bổ sung kali giúp cứng cây và hạn chế tốc độ cháy lan lên lá cờ.",
        ],
        "prevention": [
            "Luân canh ngô với cây họ đậu hoặc cây trồng khác họ ít nhất 1–2 vụ.",
            "Cày vùi hoặc thu dọn tàn dư thân lá ngô sau thu hoạch để giảm bào tử tồn lưu.",
            "Gieo trồng giống ngô lai có tính kháng bệnh đốm lá xám.",
        ],
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "medium",
        "summary": "Bệnh gỉ sắt thường trên ngô do nấm Puccinia sorghi tạo ổ bào tử màu nâu gỉ nổi rải rác hai mặt lá.",
        "treatment": [
            "Theo dõi mật độ ổ bệnh; nếu tỷ lệ lá bệnh vượt 15–20% trước giai đoạn làm hạt, phun Azoxystrobin hoặc Propiconazole.",
            "Tăng cường phân kali và vi lượng để lá giữ màu xanh bền khi vào chắc hạt.",
        ],
        "prevention": [
            "Sử dụng giống ngô lai kháng gỉ sắt phù hợp mùa vụ địa phương.",
            "Gieo đúng mật độ, tránh trồng quá dày làm tăng độ ẩm tiểu khí hậu.",
        ],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "high",
        "summary": "Bệnh cháy lá lớn (đốm lá lớn) trên ngô do nấm Exserohilum turcicum gây vết cháy dài hình thoi màu xám xanh.",
        "treatment": [
            "Phun thuốc trừ nấm chứa Azoxystrobin + Propiconazole hoặc Mancozeb ngay khi vết bệnh bắt đầu lan lên tầng lá giữa.",
            "Loại bỏ lá già sát gốc bị bệnh nặng để thông thoáng chân ruộng.",
        ],
        "prevention": [
            "Luân canh cây trồng khác họ và xử lý sạch tàn dư rơm rạ ngô sau thu hoạch.",
            "Chọn giống ngô kháng bệnh cháy lá lớn và đảm bảo thoát nước tốt khi mưa lớn.",
        ],
    },
    "Grape___Black_rot": {
        "severity": "high",
        "summary": "Bệnh thối đen nho do nấm Guignardia bidwellii gây đốm nâu viền đậm trên lá và làm quả teo tóp, đen cứng.",
        "treatment": [
            "Thu gom ngay chùm quả khô đen và lá bệnh nặng đem đốt hoặc chôn lấp.",
            "Phun thuốc trừ nấm chứa Mancozeb, Captan hoặc Myclobutanil định kỳ 7–10 ngày trong giai đoạn ẩm ướt.",
        ],
        "prevention": [
            "Tỉa cành tạo tán giàn cao thoáng, cắt bỏ lá che khuất chùm quả.",
            "Dọn sạch cỏ dại và quả rụng dưới gốc nho sau mỗi vụ.",
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "severity": "high",
        "summary": "Bệnh Esca (sởi đen nho) là phức hợp nấm gỗ làm lá xuất hiện sọc hoại tử dạng vằn hổ và quả có chấm tím đen.",
        "treatment": [
            "Cắt bỏ phần thân cành có mạch gỗ bị thâm đen cho đến phần gỗ khỏe, tiêu hủy ngay.",
            "Bôi keo liền sẹo có pha hoạt chất diệt nấm lên vết cắt lớn; cây suy kiệt nặng cần nhổ bỏ để tránh lây.",
        ],
        "prevention": [
            "Cắt tỉa vào ngày khô ráo, tránh mưa ẩm khiến bào tử nấm xâm nhập vết cắt.",
            "Khử trùng dụng cụ cắt tỉa giữa các gốc nho và bảo vệ vết thương cơ giới trên thân.",
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "severity": "medium",
        "summary": "Bệnh cháy lá (đốm lá Isariopsis) trên nho gây các mảng hoại tử nâu đỏ góc cạnh, làm lá rụng sớm vào cuối vụ.",
        "treatment": [
            "Phun thuốc gốc đồng (Copper hydroxide), Mancozeb hoặc Azoxystrobin khi đốm bệnh bắt đầu lan rộng.",
            "Ngắt bỏ lá gốc nhiễm nặng nhằm hạn chế bào tử bắn lên tầng lá trên.",
        ],
        "prevention": [
            "Thoát nước tốt cho vườn nho và giữ tán lá thông thoáng.",
            "Thu gom lá rụng cuối mùa để tiêu hủy nguồn nấm.",
        ],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "high",
        "summary": "Bệnh vàng lá gân xanh (HLB / Greening) trên cam quýt do vi khuẩn Liberibacter sống trong mạch rây, lây qua rầy chổng cánh.",
        "treatment": [
            "Hiện chưa có thuốc đặc trị vi khuẩn HLB; nếu cây nhiễm nặng, cần chặt bỏ và tiêu hủy ngay để bảo vệ cả vườn.",
            "Phun thuốc đặc trị trừ rầy chổng cánh (Imidacloprid, Thiamethoxam hoặc dầu khoáng) trước khi đốn bỏ cây bệnh.",
            "Với cây mới chớm bệnh cần giữ tạm: bổ sung phân hữu cơ và vi lượng kẽm, bo, magie để duy trì sức cây và quản lý chặt rầy.",
        ],
        "prevention": [
            "Sử dụng 100% cây giống sạch bệnh có chứng nhận từ vườn ươm nhà lưới.",
            "Kiểm soát triệt để rầy chổng cánh theo các đợt ra lộc non.",
            "Trồng cây chắn gió và thường xuyên kiểm tra vườn để phát hiện, loại bỏ sớm cây có triệu chứng lá vàng lốm đốm.",
        ],
    },
    "Peach___Bacterial_spot": {
        "severity": "high",
        "summary": "Bệnh đốm vi khuẩn trên đào do Xanthomonas arboricola gây đốm thấm nước, thủng lỗ trên lá và nứt sần trên quả.",
        "treatment": [
            "Phun thuốc gốc đồng nồng độ thấp hoặc chế phẩm sinh học kháng khuẩn theo hướng dẫn cho cây ăn quả.",
            "Không phun khi trời nắng gắt và cắt bỏ cành vượt nhiễm bệnh nặng.",
        ],
        "prevention": [
            "Trồng đai cây chắn gió để giảm gió cát làm xước lá tạo cửa ngõ cho vi khuẩn.",
            "Tránh bón quá nhiều đạm và tưới phun lên tán lá.",
        ],
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "high",
        "summary": "Bệnh đốm vi khuẩn trên ớt chuông do Xanthomonas campestris gây đốm nâu sẫm viền vàng, làm rụng lá và thối sần quả.",
        "treatment": [
            "Nhổ bỏ hoặc cắt tỉa các bộ phận bệnh nặng, tránh làm việc trên ruộng khi lá còn ướt sương.",
            "Phun luân phiên thuốc gốc đồng (Copper hydroxide) kết hợp Mancozeb hoặc chế phẩm sinh học Bacillus subtilis.",
        ],
        "prevention": [
            "Xử lý hạt giống bằng nước ấm 50°C trong 25 phút trước khi gieo.",
            "Luân canh ít nhất 2 năm với cây không thuộc họ cà (cà chua, ớt, khoai tây).",
            "Dùng màng phủ nông nghiệp và tưới nhỏ giọt để hạn chế bùn đất bắn lên lá.",
        ],
    },
    "Potato___Early_blight": {
        "severity": "medium",
        "summary": "Bệnh đốm vòng (cháy lá sớm) trên khoai tây do nấm Alternaria solani tạo đốm nâu có vòng đồng tâm đặc trưng.",
        "treatment": [
            "Ngắt bỏ lá già sát gốc bị đốm nặng để giảm nguồn bào tử.",
            "Phun thuốc trừ nấm chứa Chlorothalonil, Mancozeb, Azoxystrobin hoặc Difenoconazole định kỳ 7–10 ngày.",
        ],
        "prevention": [
            "Bón đầy đủ kali và lân để cây khỏe, tránh thiếu dinh dưỡng ở giai đoạn tạo củ.",
            "Luân canh cây trồng 2–3 năm và thu dọn sạch tàn dư thân lá sau thu hoạch.",
        ],
    },
    "Potato___Late_blight": {
        "severity": "high",
        "summary": "Bệnh mốc sương (sương mai) trên khoai tây do nấm Phytophthora infestans lây lan cực nhanh trong thời tiết mát ẩm, gây thối nhũn lá và củ.",
        "treatment": [
            "Phun khẩn cấp thuốc đặc trị mốc sương có tính nội hấp và tiếp xúc như Metalaxyl-M + Mancozeb, Dimethomorph hoặc Cymoxanil.",
            "Nếu ổ bệnh nặng cục bộ, tiêu hủy ngay cây bệnh và phun bao vây vùng xung quanh.",
            "Ngừng tưới nước lên tán và tháo thoát nước nhanh nếu ruộng bị úng.",
        ],
        "prevention": [
            "Dùng củ giống sạch bệnh, không lấy giống từ ruộng từng nhiễm mốc sương.",
            "Vun luống cao để tránh bào tử theo nước mưa thấm xuống làm thối củ.",
            "Phun phòng sớm khi thời tiết âm u, sương mù nhiều và độ ẩm > 85%.",
        ],
    },
    "Squash___Powdery_mildew": {
        "severity": "medium",
        "summary": "Bệnh phấn trắng trên bí do nấm Podosphaera xanthii phủ lớp phấn trắng xám hai mặt lá, làm lá vàng khô và quả nhỏ.",
        "treatment": [
            "Cắt bỏ lá già dưới gốc bị phủ phấn nặng và tiêu hủy.",
            "Phun lưu huỳnh, Potassium bicarbonate, dầu neem hoặc Azoxystrobin luân phiên.",
        ],
        "prevention": [
            "Trồng khoảng cách hợp lý, làm giàn hoặc tỉa lá già cho thông thoáng.",
            "Tưới gốc vào buổi sáng, tránh để ẩm độ cao kéo dài ban đêm.",
        ],
    },
    "Strawberry___Leaf_scorch": {
        "severity": "medium",
        "summary": "Bệnh cháy lá dâu tây do nấm Diplocarpon earlianum gây nhiều đốm tím sẫm liên kết thành mảng cháy khô mép lá.",
        "treatment": [
            "Nhặt bỏ toàn bộ lá già, lá bị đốm tím nặng ra khỏi luống dâu.",
            "Phun thuốc trừ nấm chứa Captan, Azoxystrobin hoặc gốc đồng theo liều khuyến cáo.",
        ],
        "prevention": [
            "Tưới nhỏ giọt dưới màng phủ thay vì tưới phun mưa làm bắn bào tử.",
            "Chọn ngó giống khỏe mạnh và thay luống trồng định kỳ.",
        ],
    },
    "Tomato___Bacterial_spot": {
        "severity": "high",
        "summary": "Bệnh đốm vi khuẩn trên cà chua gây đốm nhỏ thấm nước hóa đen trên lá, cuống hoa và vết ghẻ trên quả.",
        "treatment": [
            "Loại bỏ lá bệnh nặng và tuyệt đối không bấm tỉa chồi khi cây đang ướt.",
            "Phun thuốc gốc đồng phối hợp Mancozeb hoặc chế phẩm vi sinh đối kháng định kỳ 5–7 ngày khi trời mưa ẩm.",
        ],
        "prevention": [
            "Sử dụng hạt giống sạch bệnh và khử trùng giá thể ươm.",
            "Luân canh với cây họ đậu/ngô, dùng tưới nhỏ giọt và phủ luống.",
        ],
    },
    "Tomato___Early_blight": {
        "severity": "medium",
        "summary": "Bệnh đốm vòng (cháy lá sớm) trên cà chua do nấm Alternaria solani gây vết đốm tròn đồng tâm màu nâu sẫm ở lá gốc.",
        "treatment": [
            "Tỉa bỏ lá gốc bị nhiễm nặng để tạo khoảng thoáng 25–30 cm sát mặt luống.",
            "Phun Chlorothalonil, Mancozeb hoặc Azoxystrobin luân phiên cách nhau 7 ngày.",
        ],
        "prevention": [
            "Làm giàn cao, phủ rơm hoặc màng phủ nilon hạn chế đất bắn lên lá.",
            "Bón cân đối N-P-K và bổ sung canxi, kali giúp lá dày khỏe.",
        ],
    },
    "Tomato___Late_blight": {
        "severity": "high",
        "summary": "Bệnh mốc sương cà chua do Phytophthora infestans gây mảng úng nước nâu đen lan nhanh toàn bộ lá, thân và quả trong điều kiện lạnh ẩm.",
        "treatment": [
            "Phun ngay thuốc đặc trị mốc sương như Dimethomorph, Metalaxyl-M + Mancozeb hoặc Mandipropamid.",
            "Thu gom thân lá, quả thối đem chôn hủy xa ruộng và tạm dừng bón đạm.",
        ],
        "prevention": [
            "Trồng mật độ thoáng, che mưa hoặc tăng thông gió nếu trồng trong nhà màng.",
            "Phun phòng khi dự báo có sương mù, mưa phùn kéo dài.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "severity": "medium",
        "summary": "Bệnh mốc lá cà chua do nấm Passalora fulva gây đốm vàng mặt trên lá và lớp mốc nâu ô-liu mịn ở mặt dưới, phổ biến trong nhà kính ẩm.",
        "treatment": [
            "Giảm ẩm độ nhà màng (< 80%) bằng cách mở cửa thông gió và tỉa lá già phía dưới.",
            "Phun thuốc trừ nấm chứa Chlorothalonil, Copper hoặc Difenoconazole tập trung mặt dưới lá.",
        ],
        "prevention": [
            "Chọn giống cà chua kháng nấm mốc lá và tránh tưới vào chiều tối.",
            "Vệ sinh, khử trùng khung nhà kính sau mỗi vụ thu hoạch.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "medium",
        "summary": "Bệnh đốm lá Septoria trên cà chua tạo hàng loạt đốm tròn nhỏ tâm xám trắng viền nâu, làm vàng và rụng lá từ gốc lên ngọn.",
        "treatment": [
            "Cắt bỏ các lá gốc có đốm bệnh và tiêu hủy, rửa sạch tay/kéo sau thao tác.",
            "Phun Mancozeb, Chlorothalonil hoặc thuốc gốc đồng đều hai mặt lá.",
        ],
        "prevention": [
            "Phủ mặt luống và tưới gốc bằng hệ thống nhỏ giọt.",
            "Luân canh ít nhất 2 năm với cây không thuộc họ cà và diệt sạch cỏ dại họ cà.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "medium",
        "summary": "Nhện đỏ hai chấm (Tetranychus urticae) chích hút mặt dưới lá làm lá chấm trắng bạc, vàng úa và có tơ mảnh khi mật độ cao.",
        "treatment": [
            "Phun thuốc đặc trị nhện (Abamectin, Spiromesifen hoặc Bifenazate) kỹ vào mặt dưới lá; luân phiên hoạt chất sau 5–7 ngày để diệt lứa trứng mới nở.",
            "Có thể phun rửa mặt dưới lá bằng tia nước nhẹ hoặc dầu khoáng nông nghiệp khi mới chớm.",
        ],
        "prevention": [
            "Giữ ẩm độ vừa phải, tránh để cây khô hạn kéo dài trong mùa nắng nóng.",
            "Bảo vệ thiên địch (nhện bắt mồi) và làm sạch cỏ dại quanh bờ ruộng.",
        ],
    },
    "Tomato___Target_Spot": {
        "severity": "medium",
        "summary": "Bệnh đốm mục tiêu trên cà chua do nấm Corynespora cassiicola tạo đốm nâu có vòng đồng tâm và dễ gây rụng lá, đốm lõm trên quả.",
        "treatment": [
            "Tỉa bỏ lá già bị bệnh dưới gốc và nâng cao thông thoáng tán cây.",
            "Phun thuốc trừ nấm chứa Azoxystrobin, Difenoconazole hoặc Chlorothalonil.",
        ],
        "prevention": [
            "Tránh tưới phun mưa lên lá và dọn sạch tàn dư thực vật cuối vụ.",
            "Bón phân cân đối để cây duy trì sức đề kháng tốt.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "high",
        "summary": "Bệnh xoăn vàng lá cà chua (TYLCV) do virus truyền qua bọ phấn trắng Bemisia tabaci, khiến ngọn cây chùn, lá non xoăn nhỏ và vàng mép.",
        "treatment": [
            "Virus không có thuốc chữa trực tiếp; cần nhổ bỏ ngay cây nhiễm nặng cho vào túi kín tiêu hủy để tránh thành nguồn lây.",
            "Phun thuốc trừ bọ phấn trắng (Imidacloprid, Dinotefuran, Pymetrozine hoặc dầu khoáng) cho toàn vườn và khu vực xung quanh.",
            "Che lưới chắn côn trùng hoặc dùng bẫy dính vàng để giảm mật độ bọ phấn.",
        ],
        "prevention": [
            "Sử dụng giống cà chua kháng TYLCV và cây giống ươm trong nhà lưới chống bọ phấn.",
            "Dùng màng phủ phản quang xua đuổi bọ phấn trắng và làm sạch cỏ dại ký chủ.",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "high",
        "summary": "Bệnh khảm virus cà chua (ToMV) làm lá loang lổ xanh đậm - nhạt, biến dạng dạng dương xỉ và lây lan cực nhanh qua tiếp xúc cơ học.",
        "treatment": [
            "Nhổ bỏ và tiêu hủy ngay cây có triệu chứng khảm để chặn lây lan.",
            "Khử trùng tay bằng xà phòng/sữa tách béo hoặc dung dịch TSP 10% và khử trùng kéo trước khi chạm vào cây khỏe.",
        ],
        "prevention": [
            "Dùng hạt giống đã xử lý nhiệt/hóa chất từ nguồn uy tín và giống mang gen kháng ToMV.",
            "Không hút thuốc lá gần vườn cà chua (virus thuốc lá TMV có thể lây chéo) và rửa tay sạch trước khi chăm sóc cây.",
        ],
    },
}


def _title_clean(text: str) -> str:
    """Làm sạch dấu gạch dưới và viết hoa chữ cái đầu hợp lý, giữ nguyên nội dung trong ngoặc."""
    normalized = " ".join(text.replace("_", " ").split())
    words = []
    for token in normalized.split(" "):
        if not token:
            continue
        if token.startswith("("):
            words.append("(" + token[1:].capitalize() if len(token) > 1 else token)
        elif token.endswith(")") and "(" not in token:
            words.append(token[:-1].capitalize() + ")")
        else:
            words.append(token[0].upper() + token[1:])
    return " ".join(words)


def format_label(raw_label: str) -> Tuple[str, str, bool]:
    """
    Tách nhãn chuẩn PlantVillage ("Plant___Condition") thành (plant, condition, is_healthy).
    Ví dụ:
        "Apple___Cedar_apple_rust" -> ("Apple", "Cedar Apple Rust", False)
        "Tomato___healthy" -> ("Tomato", "Healthy", True)
    """
    if "___" in raw_label:
        plant_raw, cond_raw = raw_label.split("___", 1)
    else:
        plant_raw, cond_raw = raw_label, "Unknown"

    plant = _title_clean(plant_raw)
    cond_norm = " ".join(cond_raw.replace("_", " ").split())
    is_healthy = cond_norm.lower() == "healthy"
    condition = "Healthy" if is_healthy else _title_clean(cond_norm)
    return plant, condition, is_healthy


def _keyword_fallback_advisory(plant: str, condition: str, raw_label: str) -> Dict[str, Any]:
    """Fallback thông minh theo từ khóa bệnh lý nếu nhãn nằm ngoài từ điển chính."""
    key = f"{condition} {raw_label}".lower()

    if any(k in key for k in ("virus", "mosaic", "curl", "greening", "haunglongbing")):
        return {
            "severity": "high",
            "summary": f"Nhóm bệnh do virus/vi khuẩn mạch dẫn trên {plant} ({condition}), thường lây qua côn trùng chích hút hoặc tiếp xúc cơ học.",
            "treatment": [
                "Loại bỏ và tiêu hủy cây nhiễm nặng để chặn nguồn lây lan sang cây khỏe.",
                "Kiểm soát triệt để côn trùng môi giới (bọ phấn, rầy, rệp) bằng thuốc đặc trị và bẫy dính vàng.",
                "Khử trùng tay và dụng cụ cắt tỉa trước khi chuyển sang cây khác.",
            ],
            "prevention": [
                "Sử dụng cây giống sạch bệnh, ưu tiên giống kháng hoặc trồng trong nhà lưới.",
                "Vệ sinh cỏ dại quanh vườn nhằm loại bỏ ký chủ trung gian của côn trùng truyền bệnh.",
            ],
        }

    if any(k in key for k in ("late_blight", "late blight", "blight", "rot", "esca")):
        return {
            "severity": "high",
            "summary": f"Bệnh hại nghiêm trọng ({condition}) trên {plant}, có khả năng làm cháy hoặc thối mô cây nhanh khi ẩm độ cao.",
            "treatment": [
                "Cắt tỉa và tiêu hủy bộ phận bị thối/cháy nặng để giảm mật độ nguồn bệnh.",
                "Phun thuốc trừ nấm đặc trị (nhóm Mancozeb, Metalaxyl-M, Azoxystrobin hoặc Dimethomorph) theo liều khuyến cáo.",
            ],
            "prevention": [
                "Đảm bảo thoát nước tốt, tỉa tán thông thoáng và tránh tưới ướt lá vào buổi tối.",
                "Luân canh cây trồng và thu dọn sạch tàn dư thực vật sau thu hoạch.",
            ],
        }

    if any(k in key for k in ("bacterial", "spot")):
        return {
            "severity": "medium",
            "summary": f"Bệnh đốm lá/vi khuẩn ({condition}) trên {plant} gây tổn thương mô quang hợp và có thể lan theo nước mưa.",
            "treatment": [
                "Ngắt bỏ lá bị đốm nặng, tránh thao tác chăm sóc khi lá đang ướt.",
                "Phun luân phiên thuốc gốc đồng hoặc thuốc trừ nấm phổ rộng (Mancozeb, Chlorothalonil) tùy tác nhân.",
            ],
            "prevention": [
                "Tưới gốc bằng nhỏ giọt, phủ luống hạn chế bùn đất bắn lên lá.",
                "Xử lý hạt giống sạch bệnh và luân canh định kỳ.",
            ],
        }

    if any(k in key for k in ("rust", "mildew", "mold", "scab", "scorch")):
        return {
            "severity": "medium",
            "summary": f"Bệnh nấm bề mặt lá ({condition}) trên {plant}, làm giảm hiệu suất quang hợp và suy yếu tán lá nếu không xử lý sớm.",
            "treatment": [
                "Loại bỏ lá già nhiễm nặng dưới tán và tăng thông thoáng khu vực trồng.",
                "Phun thuốc trừ nấm phù hợp (lưu huỳnh, Triazole hoặc Strobilurin) tập trung đều hai mặt lá.",
            ],
            "prevention": [
                "Bón phân cân đối N-P-K, tránh bón thừa đạm làm tán lá rậm rạp dễ nhiễm nấm.",
                "Thu gom lá rụng cuối vụ và giữ khoảng cách trồng hợp lý.",
            ],
        }

    if any(k in key for k in ("mite", "spider")):
        return {
            "severity": "medium",
            "summary": f"Dịch hại chích hút ({condition}) trên {plant}, thường bùng phát trong điều kiện nắng nóng khô hạn.",
            "treatment": [
                "Phun thuốc đặc trị nhện (Abamectin, Spiromesifen hoặc dầu khoáng) tập trung mặt dưới lá.",
                "Phun lặp lại sau 5–7 ngày bằng hoạt chất khác để cắt lứa trứng mới nở.",
            ],
            "prevention": [
                "Giữ ẩm hợp lý cho vườn, tránh để cây thiếu nước kéo dài.",
                "Dọn sạch cỏ dại ký chủ và bảo tồn thiên địch tự nhiên.",
            ],
        }

    return {
        "severity": "medium",
        "summary": f"Phát hiện dấu hiệu bất thường ({condition}) trên lá {plant}. Cần theo dõi sát mức độ lan rộng trên tán cây.",
        "treatment": [
            "Cách ly hoặc cắt bỏ lá có vết bệnh rõ rệt để hạn chế lây lan.",
            "Tham khảo kỹ thuật viên nông nghiệp địa phương trước khi phối trộn thuốc đặc trị.",
        ],
        "prevention": [
            "Giữ vườn thông thoáng, thoát nước tốt và bón phân cân đối.",
            "Kiểm tra định kỳ toàn vườn để xử lý kịp thời khi ổ bệnh mới phát sinh.",
        ],
    }


def get_advisory(raw_label: str) -> Dict[str, Any]:
    """
    Trả về khuyến nghị nông học bằng tiếng Việt cho nhãn bệnh/khỏe mạnh.
    Cấu trúc: {"severity": "none"|"medium"|"high", "summary": str, "treatment": list[str], "prevention": list[str]}
    """
    plant, condition, is_healthy = format_label(raw_label)
    if is_healthy:
        return {
            "severity": "none",
            "summary": f"Lá {plant} hoàn toàn khỏe mạnh, sắc tố lá đồng đều và chưa ghi nhận dấu hiệu tổn thương do nấm, vi khuẩn hay virus.",
            "treatment": list(_HEALTHY_TEMPLATE_TIPS["treatment"]),
            "prevention": list(_HEALTHY_TEMPLATE_TIPS["prevention"]),
        }

    entry = _DISEASE_ADVISORIES.get(raw_label)
    if entry is not None:
        return {
            "severity": entry["severity"],
            "summary": entry["summary"],
            "treatment": list(entry["treatment"]),
            "prevention": list(entry["prevention"]),
        }

    return _keyword_fallback_advisory(plant, condition, raw_label)


# Bảng ánh xạ tiền tố tên file mẫu PlantVillage trong `data/test/`
_SAMPLE_FILENAME_PREFIXES: List[Tuple[str, str]] = [
    ("AppleCedarRust", "Apple___Cedar_apple_rust"),
    ("AppleScab", "Apple___Apple_scab"),
    ("AppleBlackRot", "Apple___Black_rot"),
    ("AppleHealthy", "Apple___healthy"),
    ("CornCommonRust", "Corn_(maize)___Common_rust_"),
    ("CornNorthernLeafBlight", "Corn_(maize)___Northern_Leaf_Blight"),
    ("CornGrayLeafSpot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"),
    ("CornHealthy", "Corn_(maize)___healthy"),
    ("PotatoEarlyBlight", "Potato___Early_blight"),
    ("PotatoLateBlight", "Potato___Late_blight"),
    ("PotatoHealthy", "Potato___healthy"),
    ("TomatoEarlyBlight", "Tomato___Early_blight"),
    ("TomatoLateBlight", "Tomato___Late_blight"),
    ("TomatoHealthy", "Tomato___healthy"),
    ("TomatoYellowCurlVirus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
    ("TomatoMosaicVirus", "Tomato___Tomato_mosaic_virus"),
    ("TomatoBacterialSpot", "Tomato___Bacterial_spot"),
    ("TomatoLeafMold", "Tomato___Leaf_Mold"),
    ("TomatoSeptoria", "Tomato___Septoria_leaf_spot"),
    ("TomatoSpiderMites", "Tomato___Spider_mites Two-spotted_spider_mite"),
    ("TomatoTargetSpot", "Tomato___Target_Spot"),
]


def infer_sample_metadata(filename: str) -> Tuple[str, str, bool]:
    """
    Suy luận (plant, condition, is_healthy) từ tên file mẫu trong `data/test/`.
    Ví dụ: "AppleCedarRust1.JPG" -> ("Apple", "Cedar Apple Rust", False).
    """
    stem = re.sub(r"\.[^.]+$", "", filename)
    for prefix, raw_label in _SAMPLE_FILENAME_PREFIXES:
        if stem.lower().startswith(prefix.lower()):
            return format_label(raw_label)

    cleaned = re.sub(r"\d+$", "", stem)
    tokens = re.findall(r"[A-Z][a-z]*|[a-z]+", cleaned)
    if len(tokens) >= 2:
        plant = tokens[0].capitalize()
        cond_tokens = tokens[1:]
        is_healthy = len(cond_tokens) == 1 and cond_tokens[0].lower() == "healthy"
        condition = "Healthy" if is_healthy else " ".join(t.capitalize() for t in cond_tokens)
        return plant, condition, is_healthy
    return _title_clean(cleaned or stem), "Sample Leaf", False
