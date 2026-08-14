"""Benchmark the FP32 and INT8 pose ONNX models with OpenCV DNN on CPU."""
from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

import cv2
import numpy as np

import benchmark_opencv_forward as forward_benchmark


ROOT = Path(__file__).resolve().parent
DEFAULT_FP32_MODEL = ROOT / "yolo11n-pose-256.onnx"
DEFAULT_INT8_MODEL = ROOT / "yolo11n-pose-256-int8.onnx"
RESULTS_FILE = ROOT / "benchmark" / "results" / "opencv_int8_benchmark.csv"
CSV_FIELDS = (
    "model_name",
    "precision",
    "avg_forward_ms",
    "fps",
    "status",
    "error_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_FP32_MODEL)
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_INT8_MODEL)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", type=Path, default=RESULTS_FILE)
    return parser.parse_args()


def resolve_model(path: Path) -> Path:
    resolved = path if path.is_absolute() else ROOT / path
    return resolved.resolve()


def model_support(path: Path, input_size: int) -> tuple[bool, str]:
    """Check graph loading and one CPU forward before the timed benchmark."""
    if not path.exists():
        return False, f"model not found: {path}"
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setInput(np.zeros((1, 3, input_size, input_size), dtype=np.float32))
        net.forward()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def benchmark_one(
    model_path: Path,
    precision: str,
    args: argparse.Namespace,
    support: tuple[bool, str] | None = None,
) -> dict[str, str]:
    supported, error_reason = support or model_support(
        model_path, args.input_size
    )
    if not supported:
        return {
            "model_name": model_path.name,
            "precision": precision,
            "avg_forward_ms": "",
            "fps": "",
            "status": "FAILED",
            "error_reason": error_reason,
        }

    try:
        result = forward_benchmark.benchmark(
            model_path, args.input_size, args.warmup, args.iterations
        )
    except Exception as exc:
        error_reason = f"{type(exc).__name__}: {exc}"
        if precision == "INT8":
            print(f"OpenCV INT8 benchmark: FAILED - {error_reason}")
        return {
            "model_name": model_path.name,
            "precision": precision,
            "avg_forward_ms": "",
            "fps": "",
            "status": "FAILED",
            "error_reason": error_reason,
        }
    return {
        "model_name": model_path.name,
        "precision": precision,
        "avg_forward_ms": f"{result['forward']['average_ms']:.3f}",
        "fps": f"{result['forward_only_fps']:.3f}",
        "status": "SUCCESS",
        "error_reason": "",
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.input_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("input-size must be > 0, warmup >= 0, iterations > 0")
    if args.threads < 0:
        raise ValueError("threads must be >= 0")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.threads:
        cv2.setNumThreads(args.threads)

    models = (
        (resolve_model(args.model), "FP32"),
        (resolve_model(args.int8_model), "INT8"),
    )
    int8_support = model_support(models[1][0], args.input_size)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {platform.machine()} (expected LoongArch64 on target)")
    print("Backend: OpenCV DNN / Target: CPU")
    print(f"Input: {args.input_size}x{args.input_size}")
    print("Timing scope: forward() only; setInput() and blob creation excluded")
    print(
        f"OpenCV INT8 model support: "
        f"{'SUPPORTED' if int8_support[0] else 'UNSUPPORTED'}"
        + (f" - {int8_support[1]}" if int8_support[1] else "")
    )

    rows = [
        benchmark_one(models[0][0], models[0][1], args),
        benchmark_one(models[1][0], models[1][1], args, int8_support),
    ]
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {output_path}")
    return 0 if all(row["status"] == "SUCCESS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
