"""
Helmet Detection — flags people without a safety helmet in real time.

Expects a YOLO model trained with classes like: "helmet", "no_helmet" (or
"head") — adjust NO_HELMET_CLASSES below to match your model's class names.
A ready-made public option is any "hard hat detection" YOLOv8 model (search
Roboflow Universe / Ultralytics HUB for "hard hat detection dataset").

Weights are auto-resolved from runs/detect/<run_name>/weights/ — pass a
run name (e.g. helmet_train-2) via --run, or a direct path via --weights.

Usage:
    # By run name (auto-finds weights/best.pt or weights/last.pt)
    python apps/helmet_detection.py --run helmet_train-2 --source 0

    # By direct weights path (still supported)
    python apps/helmet_detection.py --weights models/helmet_yolov8.pt --source 0

    # List all available runs and their weight status
    python apps/helmet_detection.py --list_runs

    # Video file, export results + annotated video
    python apps/helmet_detection.py --run helmet_train-3 --source data/site_cam.mp4 \
        --export --save_video
"""
import argparse
import os
import sys
import time

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.video_stream import VideoStream
from common.fps_counter import FPSCounter, draw_fps
from common.exporter import ResultExporter
from common.detector import YoloDetector
from common.drawing import draw_box, draw_banner, draw_counts_panel

NO_HELMET_CLASSES = {"no_helmet", "no-helmet", "head", "person_no_helmet"}
HELMET_CLASSES = {"helmet", "hardhat", "hard_hat"}

# Base directory where Ultralytics saves training runs, and the known
# run folder names for this project.
RUNS_BASE = os.environ.get("HELMET_RUNS_BASE", r"D:\realtime-opencv-detection\runs\detect")
KNOWN_RUNS = [
    "helmet_train",
    "helmet_train-2",
    "helmet_train-3",
    "helmet_train-4",
    "helmet_train-5",
    "helmet_train-6",
]


def classify(class_name: str) -> str:
    name = class_name.lower()
    if name in NO_HELMET_CLASSES:
        return "violation"
    if name in HELMET_CLASSES:
        return "compliant"
    return "other"


def resolve_weights_for_run(run_name: str, base_dir: str = RUNS_BASE) -> str:
    """Given a run name like 'helmet_train-2', return the path to its
    best available checkpoint (best.pt preferred, last.pt as fallback)."""
    weights_dir = os.path.join(base_dir, run_name, "weights")
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"No weights folder found for run '{run_name}' at: {weights_dir}"
        )

    best_path = os.path.join(weights_dir, "best.pt")
    last_path = os.path.join(weights_dir, "last.pt")

    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        print(f"Note: 'best.pt' not found for {run_name}, using 'last.pt' instead.")
        return last_path

    raise FileNotFoundError(
        f"Run '{run_name}' has a weights/ folder but no best.pt or last.pt "
        f"(training likely didn't complete an epoch). Checked: {weights_dir}"
    )


def find_best_available_run(base_dir: str = RUNS_BASE, candidate_runs=None) -> str:
    """Scan known run folders and return the name of the first one that
    has usable weights. Useful as a default when no --run is specified."""
    candidate_runs = candidate_runs or KNOWN_RUNS
    for run_name in candidate_runs:
        weights_dir = os.path.join(base_dir, run_name, "weights")
        if os.path.exists(os.path.join(weights_dir, "best.pt")) or \
           os.path.exists(os.path.join(weights_dir, "last.pt")):
            return run_name
    raise FileNotFoundError(
        f"No run in {candidate_runs} has usable weights under {base_dir}. "
        "Train at least one run first, or pass --weights directly."
    )


def list_runs(base_dir: str = RUNS_BASE, candidate_runs=None):
    """Print the weights status of every known run — handy for a quick
    'what do I actually have trained' check before launching detection."""
    candidate_runs = candidate_runs or KNOWN_RUNS
    print(f"Scanning runs under: {base_dir}\n")
    for run_name in candidate_runs:
        weights_dir = os.path.join(base_dir, run_name, "weights")
        if not os.path.isdir(weights_dir):
            print(f"  {run_name:20s} -> NOT FOUND")
            continue
        has_best = os.path.exists(os.path.join(weights_dir, "best.pt"))
        has_last = os.path.exists(os.path.join(weights_dir, "last.pt"))
        if has_best:
            status = "OK (best.pt available)"
        elif has_last:
            status = "PARTIAL (only last.pt — training may be incomplete)"
        else:
            status = "EMPTY (no checkpoints saved)"
        print(f"  {run_name:20s} -> {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="Direct path to trained YOLO weights (.pt)")
    parser.add_argument("--run", default=None,
                         help=f"Run name to auto-resolve weights from (e.g. helmet_train-2). "
                              f"Looks in {RUNS_BASE}\\<run>\\weights\\")
    parser.add_argument("--runs_base", default=RUNS_BASE, help="Base folder containing training runs")
    parser.add_argument("--list_runs", action="store_true", help="List known runs and their weight status, then exit")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path/URL to video")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--export", action="store_true", help="Save detections to CSV/JSON")
    parser.add_argument("--save_video", action="store_true", help="Save an annotated output video")
    parser.add_argument("--alert_cooldown", type=float, default=3.0, help="Seconds between repeated violation logs for the same region")
    parser.add_argument("--display", action="store_true", default=True, help="Show live window (default on)")
    parser.add_argument("--no_display", dest="display", action="store_false", help="Run headless (no window)")
    args = parser.parse_args()

    if args.list_runs:
        list_runs(args.runs_base)
        return

    # ---------------------------
    # Resolve weights: --weights (direct) takes priority over --run;
    # if neither is given, auto-pick the first known run with usable weights.
    # ---------------------------
    if args.weights:
        weights_path = args.weights
    elif args.run:
        weights_path = resolve_weights_for_run(args.run, args.runs_base)
    else:
        auto_run = find_best_available_run(args.runs_base)
        print(f"No --weights or --run given — auto-selected run: {auto_run}")
        weights_path = resolve_weights_for_run(auto_run, args.runs_base)

    print(f"Using weights: {weights_path}")

    source = int(args.source) if args.source.isdigit() else args.source
    stream = VideoStream(source)
    props = stream.get_properties()

    detector = YoloDetector(weights_path, device=args.device, conf_thresh=args.conf)
    fps_counter = FPSCounter()

    exporter = None
    if args.export or args.save_video:
        exporter = ResultExporter(
            out_dir="results/helmet_detection",
            fieldnames=["timestamp", "frame_idx", "class", "status", "confidence", "x1", "y1", "x2", "y2"],
            save_video=args.save_video,
            video_fps=props["source_fps"] if props["is_file"] else 15,
            frame_size=(props["width"], props["height"]) if props["width"] else None,
        )

    total_frames, total_violations = 0, 0
    last_alert_time = 0

    print("Press 'q' to quit.")
    try:
        while True:
            grabbed, frame, frame_idx = stream.read()
            if not grabbed:
                if props["is_file"]:
                    print("End of video reached.")
                break

            detections = detector.detect(frame)
            frame_violations = 0

            for det in detections:
                status = classify(det.class_name)
                color = (0, 0, 255) if status == "violation" else (0, 200, 0) if status == "compliant" else (200, 200, 0)
                label = f"{det.class_name} {det.confidence:.2f}"
                draw_box(frame, det.box, label, color=color)

                if status == "violation":
                    frame_violations += 1

                if exporter and status in ("violation", "compliant"):
                    exporter.log({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "frame_idx": frame_idx,
                        "class": det.class_name,
                        "status": status,
                        "confidence": round(det.confidence, 3),
                        "x1": det.box[0], "y1": det.box[1], "x2": det.box[2], "y2": det.box[3],
                    })

            total_violations += frame_violations
            total_frames += 1

            if frame_violations > 0 and time.time() - last_alert_time > args.alert_cooldown:
                draw_banner(frame, f"⚠ {frame_violations} HELMET VIOLATION(S) DETECTED", color=(0, 0, 200))
                last_alert_time = time.time()

            fps_counter.tick()
            draw_fps(frame, fps_counter.get_fps())
            draw_counts_panel(frame, {"Total violations": total_violations})

            if exporter:
                exporter.write_frame(frame)

            if args.display:
                cv2.imshow("Helmet Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        if exporter:
            exporter.save_summary({
                "total_frames": total_frames,
                "total_violations": total_violations,
                "source": str(args.source),
                "run": args.run or "auto",
                "weights_used": weights_path,
            })
            exporter.close()
        print(f"Done. Processed {total_frames} frames, {total_violations} violation instances.")


if __name__ == "__main__":
    main()