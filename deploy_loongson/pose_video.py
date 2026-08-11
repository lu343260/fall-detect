"""Real-time camera entry point for Loongson 2K0300 deployment."""
from __future__ import annotations

import argparse
import time

import cv2

from fall_detector import FallDetector
import onnx_inference
from onnx_inference import ONNXPoseDetector


SHOW_DISPLAY = False  # Loongson headless environment: keep HighGUI disabled by default.


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX pose fall detection camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--process-every", type=int, default=3)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    show_display = SHOW_DISPLAY and not args.no_display
    process_every = max(1, args.process_every)
    model = ONNXPoseDetector("yolo11n-pose.onnx")
    print(
        f"[BOOT] loading yolo11n-pose.onnx "
        f"(input {onnx_inference.IMGSZ}x{onnx_inference.IMGSZ})"
    )
    print(
        f"({onnx_inference.IMGSZ}x{onnx_inference.IMGSZ} model; "
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
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if not args.no_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            capture_time = time.monotonic()
            annotated_frame = frame.copy()
            infer_start = time.monotonic()
            results = model(frame)
            infer_time = time.monotonic() - infer_start
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
            detector_state = fall_detector.state
            state_object = detection.state if detection is not None else detector_state
            state = (
                state_object.value
                if hasattr(state_object, "value")
                else state_object
            )
            fall = detection.fall if detection is not None else state in (1, 2)
            print(
                f"[INFER] persons={len(keypoints_xy)} "
                f"time={infer_time:.3f}s state={state} fall={fall}"
            )

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
