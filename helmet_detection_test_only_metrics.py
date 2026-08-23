"""
Helmet Detection - TEST SET ONLY Evaluation

Evaluates ONLY the test dataset. It NEVER predicts on:
    - training images
    - validation images
    - videos
    - webcam/live camera

For every training run:
    helmet_train
    helmet_train-2
    helmet_train-3
    helmet_train-4
    helmet_train-5
    helmet_train-6

the script evaluates BOTH:
    weights/best.pt
    weights/last.pt

It produces:
    1. Annotated test images
    2. Per-image detection CSV
    3. Detection metrics CSV
    4. Per-class precision/recall/F1
    5. mAP@0.50
    6. mAP@0.50:0.95
    7. Confusion matrix
    8. A combined comparison CSV

IMPORTANT:
"Accuracy" is reported as detection accuracy:
    TP / (TP + FP + FN)

This is appropriate for object-detection matching, but it is NOT
the same as ordinary image-classification accuracy because object
detection has no practical finite set of true-negative background boxes.

Requirements:
    pip install ultralytics opencv-python pandas numpy
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\realtime-opencv-detection")

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

RUN_NAMES = [
    "helmet_train",
    "helmet_train-2",
    "helmet_train-3",
    "helmet_train-4",
    "helmet_train-5",
    "helmet_train-6",
]

# TEST SET ONLY
TEST_IMAGES_DIR = PROJECT_ROOT / "helmet-dataset" / "test" / "images"
TEST_LABELS_DIR = PROJECT_ROOT / "helmet-dataset" / "test" / "labels"

# Optional dataset YAML. If this exists, Ultralytics model.val()
# will also calculate official mAP/precision/recall.
DATA_YAML = PROJECT_ROOT / "helmet-dataset" / "data.yaml"

OUTPUT_ROOT = PROJECT_ROOT / "results" / "helmet_test_only_metrics"

CONF_THRESH = 0.40
IOU_THRESH = 0.50
IMG_SIZE = 640

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES_FALLBACK = {
    0: "With Helmet",
    1: "Without Helmet",
}

CLASS_COLORS = {
    "With Helmet": (0, 200, 0),
    "Without Helmet": (0, 0, 255),
}


# ============================================================
# BASIC UTILITIES
# ============================================================

def get_class_name(names, class_id):
    """Safely obtain a class name from an Ultralytics names object."""
    if isinstance(names, dict):
        return str(names.get(class_id, CLASS_NAMES_FALLBACK.get(class_id, str(class_id))))

    if isinstance(names, list):
        if 0 <= class_id < len(names):
            return str(names[class_id])

    return CLASS_NAMES_FALLBACK.get(class_id, str(class_id))


def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    """Convert normalized YOLO xywh labels to pixel xyxy boxes."""
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h

    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y2 = max(0.0, min(float(img_h - 1), y2))

    return [x1, y1, x2, y2]


def box_iou(box_a, box_b):
    """IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def load_ground_truth(label_path, img_w, img_h):
    """
    Load YOLO-format test labels:
        class_id x_center y_center width height

    Only labels belonging to the TEST image are loaded.
    """
    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"[WARNING] Invalid label line in {label_path} "
                    f"line {line_number}: {line}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(
                    f"[WARNING] Non-numeric label in {label_path} "
                    f"line {line_number}: {line}"
                )
                continue

            box = xywhn_to_xyxy(x, y, w, h, img_w, img_h)

            ground_truth.append(
                {
                    "class_id": class_id,
                    "box": box,
                }
            )

    return ground_truth


def match_predictions_to_ground_truth(
    predictions,
    ground_truth,
    iou_threshold=0.50,
):
    """
    Greedy one-to-one matching.

    A prediction is a TP only when:
        - its class matches the GT class
        - IoU >= threshold

    Returns:
        tp
        fp
        fn
        matched_pairs
    """
    matched_gt = set()
    matched_pairs = []

    # Highest-confidence predictions first.
    predictions = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    tp = 0
    fp = 0

    for pred_index, pred in enumerate(predictions):
        best_gt_index = -1
        best_iou = 0.0

        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue

            current_iou = box_iou(pred["box"], gt["box"])

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if (
            best_gt_index >= 0
            and best_iou >= iou_threshold
            and predictions[pred_index]["class_id"]
            == ground_truth[best_gt_index]["class_id"]
        ):
            tp += 1
            matched_gt.add(best_gt_index)

            matched_pairs.append(
                {
                    "pred_index": pred_index,
                    "gt_index": best_gt_index,
                    "iou": best_iou,
                    "correct_class": True,
                }
            )
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn, matched_pairs


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1 and detection accuracy."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # Detection accuracy definition:
    # TP / (TP + FP + FN)
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }


# ============================================================
# CUSTOM TEST-SET EVALUATION
# ============================================================

def evaluate_weights(run_name, weight_name, test_images):
    """Run prediction ONLY on test images and calculate metrics."""

    weights_path = RUNS_DIR / run_name / "weights" / weight_name

    if not weights_path.exists():
        print(
            f"\n[SKIP] {run_name} / {weight_name}: "
            f"weights not found:\n{weights_path}"
        )
        return None

    print("\n" + "=" * 80)
    print(f"RUN       : {run_name}")
    print(f"WEIGHTS   : {weight_name}")
    print(f"PATH      : {weights_path}")
    print("EVALUATION: TEST SET ONLY")
    print("=" * 80)

    model = YOLO(str(weights_path))
    class_names = model.names

    run_output_dir = OUTPUT_ROOT / run_name / weight_name.replace(".pt", "")
    annotated_dir = run_output_dir / "annotated_test_images"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = run_output_dir / "test_detections.csv"
    image_metrics_csv = run_output_dir / "test_image_metrics.csv"
    class_metrics_csv = run_output_dir / "test_class_metrics.csv"

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_confidences = []
    all_ious = []

    per_class = {}

    start_time = time.time()
    processed_images = 0

    with open(
        detections_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "image",
                "prediction_class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

        image_rows = []

        for image_path in test_images:
            # ------------------------------------------------
            # READ TEST IMAGE ONLY
            # ------------------------------------------------
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[WARNING] Could not read: {image_path}")
                continue

            img_h, img_w = image.shape[:2]

            # ------------------------------------------------
            # READ TEST GROUND-TRUTH LABEL ONLY
            # ------------------------------------------------
            label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"

            ground_truth = load_ground_truth(
                label_path,
                img_w,
                img_h,
            )

            # ------------------------------------------------
            # PREDICTION: TEST IMAGE ONLY
            #
            # NO video source.
            # NO webcam.
            # NO training directory.
            # NO validation directory.
            # ------------------------------------------------
            results = model.predict(
                source=image,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                imgsz=IMG_SIZE,
                device="cpu",
                verbose=False,
            )[0]

            predictions = []

            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    x1, y1, x2, y2 = map(
                        float,
                        box.xyxy[0].tolist(),
                    )

                    predictions.append(
                        {
                            "class_id": class_id,
                            "class_name": get_class_name(
                                class_names,
                                class_id,
                            ),
                            "confidence": confidence,
                            "box": [x1, y1, x2, y2],
                        }
                    )

                    writer.writerow(
                        [
                            image_path.name,
                            get_class_name(class_names, class_id),
                            round(confidence, 6),
                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),
                        ]
                    )

                    all_confidences.append(confidence)

            # ------------------------------------------------
            # MATCH TEST PREDICTIONS TO TEST GROUND TRUTH
            # ------------------------------------------------
            tp, fp, fn, matched_pairs = match_predictions_to_ground_truth(
                predictions,
                ground_truth,
                IOU_THRESH,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            for pair in matched_pairs:
                all_ious.append(pair["iou"])

                gt_class_id = ground_truth[pair["gt_index"]]["class_id"]
                gt_class_name = get_class_name(
                    class_names,
                    gt_class_id,
                )

                if gt_class_name not in per_class:
                    per_class[gt_class_name] = {
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                    }

                per_class[gt_class_name]["TP"] += 1

            # Count false negatives by GT class.
            matched_gt_indices = {
                pair["gt_index"] for pair in matched_pairs
            }

            for gt_index, gt in enumerate(ground_truth):
                if gt_index not in matched_gt_indices:
                    gt_class_name = get_class_name(
                        class_names,
                        gt["class_id"],
                    )

                    if gt_class_name not in per_class:
                        per_class[gt_class_name] = {
                            "TP": 0,
                            "FP": 0,
                            "FN": 0,
                        }

                    per_class[gt_class_name]["FN"] += 1

            # Count false positives by predicted class.
            matched_pred_indices = {
                pair["pred_index"] for pair in matched_pairs
            }

            for pred_index, pred in enumerate(predictions):
                if pred_index not in matched_pred_indices:
                    pred_class_name = pred["class_name"]

                    if pred_class_name not in per_class:
                        per_class[pred_class_name] = {
                            "TP": 0,
                            "FP": 0,
                            "FN": 0,
                        }

                    per_class[pred_class_name]["FP"] += 1

            image_metric = calculate_metrics(
                tp,
                fp,
                fn,
            )

            image_rows.append(
                {
                    "image": image_path.name,
                    "ground_truth_objects": len(ground_truth),
                    "predicted_objects": len(predictions),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": image_metric["precision"],
                    "recall": image_metric["recall"],
                    "f1_score": image_metric["f1_score"],
                    "accuracy": image_metric["accuracy"],
                }
            )

            # ------------------------------------------------
            # DRAW TEST PREDICTIONS ONLY
            # ------------------------------------------------
            annotated = image.copy()

            for pred in predictions:
                x1, y1, x2, y2 = map(
                    int,
                    pred["box"],
                )

                class_name = pred["class_name"]
                confidence = pred["confidence"]

                color = CLASS_COLORS.get(
                    class_name,
                    (255, 255, 0),
                )

                label = f"{class_name} {confidence:.2f}"

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            cv2.imwrite(
                str(annotated_dir / image_path.name),
                annotated,
            )

            processed_images += 1

            print(
                f"[{processed_images}/{len(test_images)}] "
                f"{image_path.name}: "
                f"TP={tp}, FP={fp}, FN={fn}"
            )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # OVERALL TEST-SET METRICS
    # --------------------------------------------------------
    overall = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
    )

    avg_confidence = (
        float(np.mean(all_confidences))
        if all_confidences
        else 0.0
    )

    avg_iou = (
        float(np.mean(all_ious))
        if all_ious
        else 0.0
    )

    f2_denominator = (
        4 * overall["precision"] + overall["recall"]
    )

    f2_score = (
        5 * overall["precision"] * overall["recall"] / f2_denominator
        if f2_denominator
        else 0.0
    )

    # --------------------------------------------------------
    # SAVE IMAGE METRICS
    # --------------------------------------------------------
    pd.DataFrame(image_rows).to_csv(
        image_metrics_csv,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE PER-CLASS METRICS
    # --------------------------------------------------------
    class_rows = []

    for class_name, values in sorted(per_class.items()):
        metrics = calculate_metrics(
            values["TP"],
            values["FP"],
            values["FN"],
        )

        class_rows.append(
            {
                "class": class_name,
                "TP": values["TP"],
                "FP": values["FP"],
                "FN": values["FN"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
            }
        )

    pd.DataFrame(class_rows).to_csv(
        class_metrics_csv,
        index=False,
    )

    summary = {
        "run_name": run_name,
        "weights": weight_name,
        "weights_path": str(weights_path),
        "test_images": processed_images,
        "ground_truth_objects": total_tp + total_fn,
        "predicted_objects": total_tp + total_fp,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1_score": overall["f1_score"],
        "f2_score": f2_score,
        "accuracy": overall["accuracy"],
        "avg_iou_TP": avg_iou,
        "avg_confidence": avg_confidence,
        "mAP50": np.nan,
        "mAP50_95": np.nan,
        "elapsed_sec": elapsed,
        "sec_per_image": (
            elapsed / processed_images
            if processed_images
            else 0.0
        ),
        "annotated_test_dir": str(annotated_dir),
        "detections_csv": str(detections_csv),
        "image_metrics_csv": str(image_metrics_csv),
        "class_metrics_csv": str(class_metrics_csv),
    }

    # --------------------------------------------------------
    # OFFICIAL ULTRALYTICS TEST-SET VALIDATION
    # --------------------------------------------------------
    # This uses ONLY split="test" from DATA_YAML.
    # It does NOT run training or validation images.
    # --------------------------------------------------------
    if DATA_YAML.exists():
        print("\nCalculating official Ultralytics TEST metrics...")
        print(f"Dataset YAML: {DATA_YAML}")

        try:
            validation = model.val(
                data=str(DATA_YAML),
                split="test",
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                device="cpu",
                verbose=False,
                plots=True,
                save_json=False,
            )

            summary["mAP50"] = float(
                validation.box.map50
            )

            summary["mAP50_95"] = float(
                validation.box.map
            )

            # Official detection metrics.
            summary["official_precision"] = float(
                validation.box.mp
            )

            summary["official_recall"] = float(
                validation.box.mr
            )

            official_p = summary["official_precision"]
            official_r = summary["official_recall"]

            summary["official_f1_score"] = (
                2 * official_p * official_r / (official_p + official_r)
                if official_p + official_r
                else 0.0
            )

            print(
                f"Official Precision : {summary['official_precision']:.6f}"
            )
            print(
                f"Official Recall    : {summary['official_recall']:.6f}"
            )
            print(
                f"Official F1        : {summary['official_f1_score']:.6f}"
            )
            print(
                f"mAP@0.50           : {summary['mAP50']:.6f}"
            )
            print(
                f"mAP@0.50:0.95      : {summary['mAP50_95']:.6f}"
            )

        except Exception as exc:
            print(
                "[WARNING] Official model.val(test) failed:"
            )
            print(exc)

    # --------------------------------------------------------
    # SAVE SINGLE METRICS CSV
    # --------------------------------------------------------
    metrics_csv = run_output_dir / "metrics.csv"

    pd.DataFrame([summary]).to_csv(
        metrics_csv,
        index=False,
    )

    print("\n--- TEST-SET RESULTS ---")
    print(f"Test images       : {summary['test_images']}")
    print(f"Ground-truth objs : {summary['ground_truth_objects']}")
    print(f"Predicted objs    : {summary['predicted_objects']}")
    print(f"TP                : {summary['TP']}")
    print(f"FP                : {summary['FP']}")
    print(f"FN                : {summary['FN']}")
    print(f"Precision         : {summary['precision']:.6f}")
    print(f"Recall            : {summary['recall']:.6f}")
    print(f"F1-score          : {summary['f1_score']:.6f}")
    print(f"F2-score          : {summary['f2_score']:.6f}")
    print(f"Accuracy*         : {summary['accuracy']:.6f}")
    print(f"Average IoU (TP)  : {summary['avg_iou_TP']:.6f}")
    print(f"Average confidence: {summary['avg_confidence']:.6f}")
    print(f"Time              : {elapsed:.2f} sec")
    print(
        f"Time/image        : "
        f"{summary['sec_per_image']:.4f} sec"
    )

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {class_metrics_csv}")

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("HELMET DETECTION - TEST SET ONLY")
    print("=" * 80)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Test images  : {TEST_IMAGES_DIR}")
    print(f"Test labels  : {TEST_LABELS_DIR}")
    print(f"Output       : {OUTPUT_ROOT}")

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"TEST IMAGES DIRECTORY NOT FOUND:\n{TEST_IMAGES_DIR}"
        )

    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(
            f"TEST LABELS DIRECTORY NOT FOUND:\n{TEST_LABELS_DIR}\n\n"
            "Precision/recall/F1/accuracy require ground-truth test labels."
        )

    test_images = sorted(
        [
            p
            for p in TEST_IMAGES_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not test_images:
        raise RuntimeError(
            f"No test images found in:\n{TEST_IMAGES_DIR}"
        )

    print(f"\nTEST IMAGES FOUND: {len(test_images)}")
    print("\nIMPORTANT:")
    print("Predictions will be made ONLY on these test images.")
    print("No videos will be processed.")
    print("No training images will be processed.")
    print("No validation images will be processed.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Evaluate BOTH best.pt and last.pt for every run.
    for run_name in RUN_NAMES:
        for weight_name in ("best.pt", "last.pt"):
            result = evaluate_weights(
                run_name,
                weight_name,
                test_images,
            )

            if result is not None:
                all_results.append(result)

    # --------------------------------------------------------
    # COMBINED COMPARISON
    # --------------------------------------------------------
    if not all_results:
        raise RuntimeError(
            "No best.pt or last.pt weights were found."
        )

    comparison = pd.DataFrame(all_results)

    # Rank primarily by F1, then mAP50.
    comparison["ranking_f1"] = comparison["f1_score"].fillna(-1)
    comparison["ranking_map50"] = comparison["mAP50"].fillna(-1)

    comparison = comparison.sort_values(
        by=["ranking_f1", "ranking_map50"],
        ascending=False,
    )

    comparison_csv = OUTPUT_ROOT / "ALL_TEST_MODELS_COMPARISON.csv"

    comparison.to_csv(
        comparison_csv,
        index=False,
    )

    # --------------------------------------------------------
    # CLEAN DISPLAY
    # --------------------------------------------------------
    display_columns = [
        "run_name",
        "weights",
        "test_images",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "mAP50",
        "mAP50_95",
        "avg_iou_TP",
        "avg_confidence",
        "sec_per_image",
    ]

    print("\n\n" + "=" * 120)
    print("FINAL COMPARISON - TEST SET ONLY")
    print("=" * 120)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        250,
        "display.precision",
        4,
    ):
        print(comparison[display_columns].to_string(index=False))

    best_model = comparison.iloc[0]

    print("\n" + "=" * 80)
    print("BEST TEST-SET MODEL")
    print("=" * 80)
    print(
        f"Run       : {best_model['run_name']}"
    )
    print(
        f"Weights   : {best_model['weights']}"
    )
    print(
        f"Precision : {best_model['precision']:.6f}"
    )
    print(
        f"Recall    : {best_model['recall']:.6f}"
    )
    print(
        f"F1-score  : {best_model['f1_score']:.6f}"
    )
    print(
        f"Accuracy* : {best_model['accuracy']:.6f}"
    )
    print(
        f"mAP50     : {best_model['mAP50']:.6f}"
    )
    print(
        f"mAP50-95  : {best_model['mAP50_95']:.6f}"
    )

    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(f"Comparison CSV: {comparison_csv}")
    print(
        f"Per-model results: "
        f"{OUTPUT_ROOT}\\<run>\\<best_or_last>\\"
    )
    print("\nDONE.")
    print(
        "\n* Accuracy = TP / (TP + FP + FN), "
        "because ordinary TN is not well-defined for object detection."
    )


if __name__ == "__main__":
    main()
