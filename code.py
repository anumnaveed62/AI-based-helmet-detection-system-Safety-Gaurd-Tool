from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="D:/realtime-opencv-detection/helmet-dataset/data.yaml",
    epochs=30,
    patience=10,
    imgsz=416,
    batch=8,
    name="helmet_train",
    exist_ok=True,
    device="cpu"
)