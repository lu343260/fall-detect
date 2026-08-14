"""Benchmark real camera/video inference with YOLO frame skipping."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from fall_detector import FallDetector
from onnx_inference import ONNXPoseDetector

FRAME_SKIPS = (1, 2, 5, 10)


def run_one(model, source, frame_skip, duration, no_rotate, model_name):
    detector = FallDetector()
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    start = time.perf_counter()
    frames = inferences = 0
    inference_times = []
    last_detection = None
    try:
        while time.perf_counter() - start < duration:
            ret, frame = video.read()
            if not ret:
                break
            frames += 1
            if not no_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frames % frame_skip != 0:
                continue
            infer_start = time.perf_counter()
            results = model(frame)
            inference_times.append((time.perf_counter() - infer_start) * 1000)
            inferences += 1
            result = results[0]
            if len(result.keypoints.xy) > 0:
                person = result.keypoints.xy[0]
                confidence = result.keypoints.conf[0]
                if (frames % 3 == 0 and
                        all(confidence[i] >= 0.5 for i in (5, 6, 11, 12))):
                    last_detection = detector.detect(person, time.monotonic())
    finally:
        video.release()
    elapsed = max(time.perf_counter() - start, 1e-9)
    state = getattr(getattr(last_detection, "state", detector.state), "name", str(detector.state))
    return {
        "model_name": Path(model_name).name,
        "input_size": f"{model.imgsz}x{model.imgsz}",
        "frame_skip": frame_skip,
        "total_frames": frames,
        "inference_count": inferences,
        "avg_inference_time_ms": f"{sum(inference_times) / len(inference_times):.2f}" if inference_times else "0.00",
        "camera_fps": f"{frames / elapsed:.2f}",
        "ai_fps": f"{inferences / elapsed:.2f}",
        "fall_detection_result": state,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--model", default="yolo11n-pose-256.onnx")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/frame_skip_benchmark.csv"))
    parser.add_argument("--no-rotate", action="store_true")
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    model = ONNXPoseDetector(args.model, imgsz=256)
    rows = [run_one(model, source, skip, args.duration, args.no_rotate, args.model) for skip in FRAME_SKIPS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("model_name", "input_size", "frame_skip", "total_frames", "inference_count",
              "avg_inference_time_ms", "camera_fps", "ai_fps", "fall_detection_result")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
