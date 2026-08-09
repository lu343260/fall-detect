"""Real-time camera entry point for Loongson 2K0300 deployment."""
from __future__ import annotations

import argparse
import time

import cv2

from fall_detector import FallDetector
from onnx_inference import ONNXPoseDetector


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
    process_every = max(1, args.process_every)
    model = ONNXPoseDetector("yolo11n-pose.onnx")
    fall_detector = FallDetector()
    video = cv2.VideoCapture(args.camera)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not video.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    if not args.no_display:
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
            results = model(frame)
            result = results[0]
            keypoints_xy = result.keypoints.xy
            keypoints_conf = result.keypoints.conf

            if len(keypoints_xy) > 0:
                annotated_frame = result.plot()
                person = keypoints_xy[0]
                confidence = keypoints_conf[0]
                required_points = [5, 6, 11, 12]
                points_valid = all(confidence[i] >= 0.5 for i in required_points)
            else:
                person = None
                points_valid = False

            if points_valid and frame_count % process_every == 0:
                detection = fall_detector.detect(person, capture_time)
                print(f"state={detection.state.value} fall={detection.fall}")

            if not args.no_display:
                cv2.imshow("ONNX Pose", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        video.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
