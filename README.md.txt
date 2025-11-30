PHỤ LỤC 3: HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG
1. Link Repository (GitHub) Toàn bộ mã nguồn dự án, bao gồm mã huấn luyện và ứng dụng demo, được lưu trữ công khai tại:
•	[https://github.com/hiep-dev/LicensePlateRecognition_YOLOv4] 

2. Yêu cầu hệ thống Để chạy ứng dụng demo trên máy tính cá nhân, cần đáp ứng các yêu cầu sau:
•	Hệ điều hành: Windows 10/11, macOS hoặc Linux.
•	Ngôn ngữ lập trình: Python 3.8 trở lên (Khuyến nghị Python 3.12).
•	Các thư viện Python cần thiết: OpenCV (opencv-python), NumPy, Pillow (PIL), Tkinter (tích hợp sẵn trong Python).
3. Quy trình cài đặt
Bước 1: Tải mã nguồn và Dữ liệu mô hình Tải thư mục dự án về máy tính. Đảm bảo đã tải đủ các file trọng số (.weights) và cấu hình (.cfg) từ quá trình huấn luyện trên Google Drive về máy.
Bước 2: Cài đặt thư viện Mở Command Prompt (CMD) hoặc Terminal tại thư mục dự án và chạy lệnh sau để cài đặt các gói phụ thuộc:
pip install opencv-python numpy pillow
Bước 3: Kiểm tra cấu trúc thư mục Để ứng dụng hoạt động chính xác, các tệp tin trong thư mục dự án phải được sắp xếp theo đúng cấu trúc sau (đặc biệt lưu ý tên file cấu hình và trọng số):
/LicensePlateApp/
│── app.py                       # File mã nguồn chính (Giao diện Tkinter)
│── yolov4-tiny-obj.cfg          # File cấu hình mạng nơ-ron (Tải từ Colab)
│── yolov4-tiny-obj_last.weights # File trọng số đã huấn luyện (Tải từ Drive)
│── obj.names                    # File chứa tên nhãn (Nội dung: license_plate)
│── test_images/                 # (Tùy chọn) Thư mục chứa các ảnh xe để test
│    └── xe_01.jpg
Bước 4: Khởi chạy ứng dụng Tại giao diện dòng lệnh (CMD) trong thư mục dự án, chạy lệnh sau để bật phần mềm:
python app.py
Bước 5: Hướng dẫn sử dụng
1.	Giao diện phần mềm "HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE" sẽ hiện ra.
2.	Nhấn vào nút "📂 CHỌN ẢNH NGAY".
3.	Cửa sổ chọn file hiện ra, tìm và chọn một bức ảnh xe máy hoặc ô tô (định dạng .jpg, .png).
4.	Hệ thống sẽ tự động xử lý và hiển thị kết quả:
o	Hình ảnh: Vẽ khung màu xanh bao quanh biển số.
o	Nhãn: Hiển thị độ tin cậy (Confidence score) cạnh khung bao.
o	Trạng thái: Thông báo số lượng biển số tìm thấy ở dòng trạng thái phía dưới.
