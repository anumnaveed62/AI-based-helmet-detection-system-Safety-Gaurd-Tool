"""
Run trained helmet detection models on a test dataset, across ALL training runs
(helmet_train, helmet_train-2, ... helmet_train-6), so results can be compared
side by side to see which run produced the best model.

For each run, uses that run's weights\best.pt (classes: "With Helmet", "Without
Helmet") to detect and classify helmet usage across all images in the test
folder, saving annotated outputs and a detections CSV per run, plus a combined
comparison summary across all runs at the end.

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
PROJECT_ROOT = r"D:\realtime-opencv-detection"
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "detect")

# All training runs to evaluate and compare.
RUN_NAMES = [
    "helmet_train",
    "helmet_train-2",
    "helmet_train-3",
    "helmet_train-4",
    "helmet_train-5",
    "helmet_train-6",
]

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "helmet-dataset", "test", "images")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "results", "helmet_test_detection")
CONF_THRESH = 0.4
IOU_THRESH = 0.5
IMG_SIZE = 640

CLASS_COLORS = {
    "With Helmet": (0, 200, 0),      # green = compliant
    "Without Helmet": (0, 0, 255),   # red = violation
}


# ---------------------------
# Run detection for a single training run's weights
# ---------------------------
def run_detection_for(run_name: str, test_images):
    weights_path = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")

    if not os.path.exists(weights_path):
        print(f"\n[SKIP] {run_name}: no weights found at {weights_path}")
        return None

    print(f"\n=== Running detection for: {run_name} ===")
    print(f"Weights: {weights_path}")

    model = YOLO(weights_path)
    class_names = model.names
    print(f"Classes: {class_names}")

    run_output_dir = os.path.join(OUTPUT_ROOT, run_name)
    annotated_dir = os.path.join(run_output_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    csv_path = os.path.join(run_output_dir, "detections.csv")
    summary = {
        "run_name": run_name,
        "With Helmet": 0,
        "Without Helmet": 0,
        "total_images": 0,
        "images_with_violations": 0,
        "avg_confidence": 0.0,
    }
    all_confidences = []

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
                all_confidences.append(conf)

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

    summary["avg_confidence"] = round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else 0.0
    summary["elapsed_sec"] = round(elapsed, 1)
    summary["sec_per_image"] = round(elapsed / max(summary["total_images"], 1), 3)
    summary["annotated_dir"] = annotated_dir
    summary["csv_path"] = csv_path

    print(f"Images processed: {summary['total_images']}")
    print(f"'With Helmet' detections: {summary['With Helmet']}")
    print(f"'Without Helmet' detections: {summary['Without Helmet']}")
    print(f"Images with violations: {summary['images_with_violations']}")
    print(f"Avg confidence: {summary['avg_confidence']}")
    print(f"Time: {summary['elapsed_sec']}s ({summary['sec_per_image']}s/image)")

    return summary


# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    test_images = sorted([
        f for f in Path(TEST_IMAGES_DIR).iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    print(f"Found {len(test_images)} test images in {TEST_IMAGES_DIR}")

    all_summaries = []
    for run_name in RUN_NAMES:
        summary = run_detection_for(run_name, test_images)
        if summary is not None:
            all_summaries.append(summary)

    # ---------------------------
    # Combined comparison across all runs
    # ---------------------------
    print("\n\n=========== COMPARISON ACROSS ALL RUNS ===========")
    header = f"{'Run':<16} {'Images':>7} {'With Helmet':>12} {'Without Helmet':>15} {'Violations':>11} {'AvgConf':>8} {'s/img':>7}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['run_name']:<16} {s['total_images']:>7} {s['With Helmet']:>12} "
            f"{s['Without Helmet']:>15} {s['images_with_violations']:>11} "
            f"{s['avg_confidence']:>8} {s['sec_per_image']:>7}"
        )

    comparison_csv = os.path.join(OUTPUT_ROOT, "runs_comparison.csv")
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name", "total_images", "with_helmet", "without_helmet",
            "images_with_violations", "avg_confidence", "sec_per_image", "annotated_dir", "csv_path",
        ])
        for s in all_summaries:
            writer.writerow([
                s["run_name"], s["total_images"], s["With Helmet"], s["Without Helmet"],
                s["images_with_violations"], s["avg_confidence"], s["sec_per_image"],
                s["annotated_dir"], s["csv_path"],
            ])

    print(f"\nPer-run annotated images + CSVs saved under: {OUTPUT_ROOT}\\<run_name>\\")
    print(f"Combined comparison CSV saved to: {comparison_csv}")

    if all_summaries:
        best = max(all_summaries, key=lambda s: s["avg_confidence"])
        print(f"\nHighest average confidence: {best['run_name']} ({best['avg_confidence']})")
        print("Note: avg confidence alone isn't a full quality measure — cross-check against")
        print("mAP/precision/recall from each run's own training results.csv before picking a winner.")


if __name__ == "__main__":
    main()
