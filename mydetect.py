from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("best.pt")

# 选择带检测的图像
model.predict("1197.jpg", save=True, imgsz=320, conf=0.5)# 1336 1347 1350；  1056 1111 1121 1197