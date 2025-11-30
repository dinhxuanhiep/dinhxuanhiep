import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# --- CẤU HÌNH ---
CONFIG_PATH = 'yolov4-tiny-obj.cfg'
WEIGHTS_PATH = 'yolov4-tiny-obj_last.weights'
NAMES_PATH = 'obj.names'

# Kiểm tra file
if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
    print("❌ LỖI: Thiếu file Config hoặc Weights. Hãy kiểm tra lại thư mục!")
    input("Nhấn Enter để thoát...")
    exit()

print("⏳ Đang tải mô hình AI...")
try:
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("✅ Tải mô hình thành công!")
except Exception as e:
    print(f"❌ LỖI TẢI MÔ HÌNH: {e}")
    input("Nhấn Enter để thoát...")
    exit()

def detect_image(img_path):
    # --- SỬA LỖI ĐỌC TÊN FILE TIẾNG VIỆT ---
    # Thay vì dùng cv2.imread, ta dùng numpy để đọc raw data rồi decode
    try:
        img_array = np.fromfile(img_path, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return None

    if frame is None:
        print("❌ Lỗi: Không đọc được ảnh. Có thể file bị hỏng.")
        return None

    height, width, channels = frame.shape

    # Chuẩn hóa ảnh
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Quét kết quả
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Hạ ngưỡng tin cậy xuống 0.1 để dễ bắt hơn
            if confidence > 0.1: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Khử trùng lặp (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    if len(indexes) > 0:
        for i in indexes.flatten():
            count += 1
            x, y, w, h = boxes[i]
            label = f"{int(confidences[i]*100)}%"
            
            # Vẽ khung xanh lá đậm
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Nền chữ
            cv2.rectangle(frame, (x, y-30), (x+100, y), (0, 255, 0), -1)
            # Chữ trắng
            cv2.putText(frame, label, (x, y-10), font, 0.8, (255, 255, 255), 2)
            
    return frame, count

# --- GIAO DIỆN ---
def select_file():
    file_path = filedialog.askopenfilename()
    if len(file_path) > 0:
        lbl_status.config(text=f"Đang xử lý: {os.path.basename(file_path)}...", fg="blue")
        root.update() # Cập nhật giao diện ngay lập tức
        
        try:
            result = detect_image(file_path)
            
            if result is None:
                lbl_status.config(text="❌ Lỗi: Không đọc được ảnh!", fg="red")
                return

            result_img, count = result
            
            # Chuyển màu để hiển thị lên App
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(result_img)
            
            # Resize hiển thị thông minh (giữ tỉ lệ)
            base_width = 800
            w_percent = (base_width / float(img_pil.size[0]))
            h_size = int((float(img_pil.size[1]) * float(w_percent)))
            img_pil = img_pil.resize((base_width, h_size), Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img_pil)
            panel.configure(image=img_tk)
            panel.image = img_tk
            
            if count > 0:
                lbl_status.config(text=f"✅ Tìm thấy {count} biển số!", fg="green")
            else:
                lbl_status.config(text="⚠️ Không tìm thấy biển số nào (Thử ảnh khác xem)", fg="orange")
                
        except Exception as e:
            lbl_status.config(text=f"❌ Lỗi hệ thống: {e}", fg="red")
            print(e)

root = tk.Tk()
root.title("Nhận Diện Biển Số Xe")
root.geometry("900x750")

lbl_title = Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE", font=("Arial", 22, "bold"), fg="#0066cc")
lbl_title.pack(pady=15)

btn_select = Button(root, text="📂 CHỌN ẢNH NGAY", command=select_file, font=("Arial", 14, "bold"), bg="#28a745", fg="white", padx=20, pady=10)
btn_select.pack(pady=10)

lbl_status = Label(root, text="Sẵn sàng...", font=("Arial", 12))
lbl_status.pack()

panel = Label(root, bg="#f0f0f0")
panel.pack(padx=10, pady=10, expand=True)

root.mainloop()