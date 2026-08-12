"""Real-time camera entry point for Loongson 2K0300 deployment."""
from __future__ import annotations

import argparse
import time

import cv2

from fall_detector import FallDetector
import onnx_inference
from onnx_inference import ONNXPoseDetector


SHOW_DISPLAY = False  # Loongson headless environment: keep HighGUI disabled by default.
PERFORMANCE_LOG_INTERVAL_SECONDS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX pose fall detection camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--model", default="yolo11n-pose-256.onnx",
        help="ONNX model filename/path, e.g. yolo11n-pose-256-int8.onnx",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--process-every", type=int, default=3)
    parser.add_argument(
        "--dnn-threads", type=int, default=0,
        help="OpenCV CPU threads; 0 lets OpenCV choose the default",
    )
    parser.add_argument(
        "--dnn-backend", choices=sorted(onnx_inference.BACKENDS), default="opencv",
        help="OpenCV DNN backend",
    )
    parser.add_argument(
        "--dnn-target", choices=sorted(onnx_inference.TARGETS), default="cpu",
        help="OpenCV DNN target",
    )
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    show_display = SHOW_DISPLAY and not args.no_display
    process_every = max(1, args.process_every)
    model = ONNXPoseDetector(
        args.model,
        imgsz=onnx_inference.IMGSZ,
        threads=args.dnn_threads,
        backend=args.dnn_backend,
        target=args.dnn_target,
    )
    print(
        f"[BOOT] loading {args.model} "
        f"(configured input {model.imgsz}x{model.imgsz})"
    )
    print(
        f"(ONNX input shape {model.input_shape}; "
        "Ctrl-C to abort if too slow)"
    )
    fall_detector = FallDetector()
    video = cv2.VideoCapture(args.camera)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not video.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    if show_display:
        cv2.namedWindow("ONNX Pose", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ONNX Pose", args.width, args.height)

    frame_count = 0
    report_start_time = time.monotonic()
    last_report_time = report_start_time
    report_camera_frames = 0
    report_inference_count = 0
    report_inference_times_ms = []
    report_state_skipped_frames = 0
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            report_camera_frames += 1
            if not args.no_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            capture_time = time.monotonic()
            annotated_frame = frame.copy()
            infer_start = time.monotonic()
            results = model(frame)
            infer_time = time.monotonic() - infer_start
            infer_time_ms = infer_time * 1000
            report_inference_count += 1
            report_inference_times_ms.append(infer_time_ms)
            result = results[0]
            keypoints_xy = result.keypoints.xy
            keypoints_conf = result.keypoints.conf

            if len(keypoints_xy) > 0:
                if show_display:
                    annotated_frame = result.plot()
                person = keypoints_xy[0]
                confidence = keypoints_conf[0]
                required_points = [5, 6, 11, 12]
                points_valid = all(confidence[i] >= 0.5 for i in required_points)
            else:
                person = None
                points_valid = False

            detection = None
            if points_valid and frame_count % process_every == 0:
                detection = fall_detector.detect(person, capture_time)
            elif points_valid:
                report_state_skipped_frames += 1
            detector_state = fall_detector.state
            state_object = detection.state if detection is not None else detector_state
            state = (
                state_object.value
                if hasattr(state_object, "value")
                else state_object
            )
            fall = detection.fall if detection is not None else state in (1, 2)
            now = time.monotonic()
            report_elapsed = now - last_report_time
            if report_elapsed >= PERFORMANCE_LOG_INTERVAL_SECONDS:
                camera_fps = report_camera_frames / report_elapsed
                inference_fps = report_inference_count / report_elapsed
                average_infer_time_ms = (
                    sum(report_inference_times_ms) / len(report_inference_times_ms)
                    if report_inference_times_ms else 0.0
                )
                state_skip_ratio = report_state_skipped_frames / report_camera_frames
                state_name = getattr(state_object, "name", str(state_object))
                print(
                    f"[PERF] runtime={now - report_start_time:.1f}s "
                    f"camera_fps={camera_fps:.2f} "
                    f"inference_fps={inference_fps:.2f} "
                    f"inference_ms={infer_time_ms:.2f} "
                    f"avg_inference_ms={average_infer_time_ms:.2f} "
                    f"state={state_name} "
                    f"state_skip_frames={report_state_skipped_frames} "
                    f"state_skip_ratio={state_skip_ratio:.2%}"
                )
                last_report_time = now
                report_camera_frames = 0
                report_inference_count = 0
                report_inference_times_ms.clear()
                report_state_skipped_frames = 0

            if show_display:
                cv2.imshow("ONNX Pose", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        video.release()
        if show_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
