"""Benchmark the deployed OpenCV DNN inference boundary.

Only ``setInput()`` and ``forward()`` are measured. Camera capture, image
preprocessing, display, and postprocessing are intentionally excluded.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo11n-pose-256.onnx"
DEFAULT_INPUT_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def resolve_model(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"model not found: {resolved.resolve()}")
    return resolved


def summarize(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "average_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def benchmark(model_path: Path, input_size: int, warmup: int, iterations: int):
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    input_shape = (1, 3, input_size, input_size)
    blob = np.zeros(input_shape, dtype=np.float32)

    # Warmup includes both calls, matching the measured call sequence.
    for _ in range(warmup):
        net.setInput(blob)
        net.forward()

    set_input_ms: list[float] = []
    forward_ms: list[float] = []
    output_shape = None
    for _ in range(iterations):
        start = time.perf_counter()
        net.setInput(blob)
        set_input_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        output = net.forward()
        forward_ms.append((time.perf_counter() - start) * 1000.0)
        output_shape = tuple(np.asarray(output).shape)

    set_stats = summarize(set_input_ms)
    forward_stats = summarize(forward_ms)
    total_ms = np.asarray(set_input_ms) + np.asarray(forward_ms)
    total_stats = summarize(total_ms.tolist())
    return {
        "model_name": model_path.name,
        "model_path": str(model_path.resolve()),
        "input_size": f"{input_size}x{input_size}",
        "input_shape": input_shape,
        "inference_count": iterations,
        "warmup_count": warmup,
        "setInput": set_stats,
        "forward": forward_stats,
        "setInput_plus_forward": total_stats,
        "theoretical_fps": 1000.0 / total_stats["average_ms"],
        "forward_only_fps": 1000.0 / forward_stats["average_ms"],
        "output_shape": output_shape,
    }


def print_stats(result: dict):
    print(f"model_name: {result['model_name']}")
    print(f"input_size: {result['input_size']} ({result['input_shape']})")
    print(f"inference_count: {result['inference_count']}")
    print(f"warmup_count: {result['warmup_count']}")
    for name in ("setInput", "forward", "setInput_plus_forward"):
        stats = result[name]
    print(
        f"{name}: average={stats['average_ms']:.3f} ms, "
            f"min={stats['min_ms']:.3f} ms, max={stats['max_ms']:.3f} ms, "
            f"P50={stats['p50_ms']:.3f} ms, P95={stats['p95_ms']:.3f} ms"
        )
    print(f"forward average: {result['forward']['average_ms']:.3f} ms")
    print(f"FPS: {result['forward_only_fps']:.3f}")
    print(f"theoretical_fps (setInput+forward): {result['theoretical_fps']:.3f}")
    print(f"forward_only_fps: {result['forward_only_fps']:.3f}")
    print(f"output_shape: {result['output_shape']}")


def main() -> int:
    args = parse_args()
    if (
        args.input_size <= 0
        or args.warmup < 0
        or args.iterations <= 0
        or args.threads < 0
    ):
        raise ValueError(
            "input-size must be > 0, warmup >= 0, iterations > 0, threads >= 0"
        )

    if args.threads != 0:
        cv2.setNumThreads(args.threads)

    model_path = resolve_model(args.model)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {platform.machine()} (expected LoongArch64 on target)")
    print(
        "OpenCV threads: "
        f"{args.threads if args.threads != 0 else 'default'}"
    )
    print("Backend: OpenCV DNN / Target: CPU")
    print("Timing scope: setInput() and forward() only")
    print("Input blob: fixed FP32 zeros; creation is outside timing")
    print_stats(benchmark(model_path, args.input_size, args.warmup, args.iterations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
