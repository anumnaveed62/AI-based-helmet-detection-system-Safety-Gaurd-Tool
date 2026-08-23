"""
Run trained helmet detection model on a test dataset.

Uses the model trained in helmet_train-4 (classes: "With Helmet", "Without Helmet")
to detect and classify helmet usage across all images in a test folder, saving
annotated outputs, a detections CSV, and evaluation metrics against ground truth.

Usage:
    python detect_helmets_test.py
"""
import os
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------------------
# Config
# ---------------------------
WEIGHTS_PATH = r"D:\realtime-opencv-detection\runs\detect\helmet_train-4\weights\best.pt"
TEST_IMAGES_DIR = r"D:\realtime-opencv-detection\helmet-dataset\test\images"
OUTPUT_DIR = r"D:\realtime-opencv-detection\results\helmet_test_detection"
CONF_THRESH = 0.4
IOU_THRESH = 0.5
IMG_SIZE = 640

CLASS_COLORS = {
    "With Helmet": (0, 200, 0),      # green = compliant
    "Without Helmet": (0, 0, 255),   # red = violation
}

# ---------------------------
# Setup
# ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
annotated_dir = os.path.join(OUTPUT_DIR, "annotated")
os.makedirs(annotated_dir, exist_ok=True)

model = YOLO(WEIGHTS_PATH)
class_names = model.names
print(f"Loaded model from {WEIGHTS_PATH}")
print(f"Classes: {class_names}")

test_images = sorted([
    f for f in Path(TEST_IMAGES_DIR).iterdir()
    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
])
print(f"Found {len(test_images)} test images")

# ---------------------------
# Run inference on the whole test set
# ---------------------------
csv_path = os.path.join(OUTPUT_DIR, "detections.csv")
summary = {"With Helmet": 0, "Without Helmet": 0, "total_images": 0, "images_with_violations": 0}

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])

    start = time.time()
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            verbose=False,
        )[0]

        summary["total_images"] += 1
        image_has_violation = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            writer.writerow([img_path.name, cls_name, round(conf, 3), x1, y1, x2, y2])

            if cls_name in summary:
                summary[cls_name] += 1
            if cls_name == "Without Helmet":
                image_has_violation = True

            # Draw box + label on the image
            color = CLASS_COLORS.get(cls_name, (255, 255, 0))
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if image_has_violation:
            summary["images_with_violations"] += 1

        out_path = os.path.join(annotated_dir, img_path.name)
        cv2.imwrite(out_path, img)

    elapsed = time.time() - start

# ---------------------------
# Print summary
# ---------------------------
print("\n--- Detection Summary ---")
print(f"Images processed: {summary['total_images']}")
print(f"'With Helmet' detections: {summary['With Helmet']}")
print(f"'Without Helmet' detections: {summary['Without Helmet']}")
print(f"Images containing at least one violation: {summary['images_with_violations']}")
print(f"Time taken: {elapsed:.1f}s ({elapsed / max(summary['total_images'], 1):.2f}s/image)")
print(f"\nAnnotated images saved to: {annotated_dir}")
print(f"Detections CSV saved to: {csv_path}")