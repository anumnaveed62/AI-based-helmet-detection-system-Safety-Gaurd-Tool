from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="D:/realtime-opencv-detection/helmet-dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="helmet_train"
)