from ultralytics import YOLO

configs = ["helmet_train-2", "helmet_train-3"]

for name in configs:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="D:/realtime-opencv-detection/helmet-dataset/data.yaml",
        epochs=30,
        patience=10,
        imgsz=416,
        batch=8,
        name=name,
        exist_ok=True,
        device="cpu"
    )
    print(f"{name} finished.\n")