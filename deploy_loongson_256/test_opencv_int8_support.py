"""Check whether LoongArch OpenCV DNN can execute the INT8 pose model once."""
from __future__ import annotations

import argparse
import platform
import traceback
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo11n-pose-256-int8.onnx"
INPUT_SHAPE = (1, 3, 256, 256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="INT8 ONNX model path (default: yolo11n-pose-256-int8.onnx)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {platform.machine()} (expected LoongArch64)")
    print(f"Model: {model_path.resolve()}")
    print(f"Input shape: {INPUT_SHAPE}")

    if not model_path.exists():
        print("Model loading: FAILED")
        print(f"FileNotFoundError: model not found: {model_path.resolve()}")
        print("Forward: NOT RUN")
        return 1

    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model loading: SUCCESS")
    except Exception:
        print("Model loading: FAILED")
        traceback.print_exc()
        print("Forward: NOT RUN")
        return 1

    try:
        # A zero-valued FP32 blob is sufficient to validate graph execution.
        blob = np.zeros(INPUT_SHAPE, dtype=np.float32)
        net.setInput(blob)
        output = net.forward()
        print("Forward: SUCCESS")
        print(f"Output shape: {tuple(np.asarray(output).shape)}")
        return 0
    except Exception:
        print("Forward: FAILED")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
