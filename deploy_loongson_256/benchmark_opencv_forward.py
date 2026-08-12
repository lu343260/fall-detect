"""Benchmark OpenCV DNN FP32/INT8 model forward time only.

This benchmark excludes camera capture, preprocessing, postprocessing, drawing,
and result handling. Only ``net.forward()`` is timed after a fixed input blob
has been assigned with ``setInput``.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_FP32 = ROOT / "yolo11n-pose-256.onnx"
DEFAULT_INT8 = ROOT / "yolo11n-pose-256-int8.onnx"
INPUT_SHAPE = (1, 3, 256, 256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, default=DEFAULT_FP32)
    parser.add_argument("--int8", type=Path, default=DEFAULT_INT8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def resolve_model(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"model not found: {resolved.resolve()}")
    return resolved


def benchmark(path: Path, warmup: int, iterations: int):
    net = cv2.dnn.readNetFromONNX(str(path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Keep input preparation outside the timed region.
    blob = np.zeros(INPUT_SHAPE, dtype=np.float32)
    net.setInput(blob)
    for _ in range(warmup):
        net.forward()

    elapsed_ms = []
    output_shape = None
    for _ in range(iterations):
        start = time.perf_counter()
        output = net.forward()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        output_shape = tuple(np.asarray(output).shape)

    average_ms = float(np.mean(elapsed_ms))
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "average_forward_ms": average_ms,
        "forward_fps": 1000.0 / average_ms if average_ms > 0 else 0.0,
        "output_shape": output_shape,
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be >= 0 and --iterations must be > 0")

    fp32_path = resolve_model(args.fp32)
    int8_path = resolve_model(args.int8)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {platform.machine()} (expected LoongArch64)")
    print("Backend: OpenCV DNN")
    print("Target: CPU")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Warmup: {args.warmup}, iterations: {args.iterations}")
    print("Timing scope: net.forward() only")

    results = {}
    for name, path in (("FP32", fp32_path), ("INT8", int8_path)):
        result = benchmark(path, args.warmup, args.iterations)
        results[name] = result
        print(
            f"{name}: average_forward={result['average_forward_ms']:.3f} ms, "
            f"forward_FPS={result['forward_fps']:.3f}, "
            f"output_shape={result['output_shape']}"
        )

    fp32_ms = results["FP32"]["average_forward_ms"]
    int8_ms = results["INT8"]["average_forward_ms"]
    speedup = fp32_ms / int8_ms if int8_ms > 0 else 0.0
    change_percent = ((fp32_ms - int8_ms) / fp32_ms * 100.0) if fp32_ms > 0 else 0.0
    print(f"INT8 speedup ratio: {speedup:.3f}x")
    print(f"INT8 forward time change: {change_percent:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
