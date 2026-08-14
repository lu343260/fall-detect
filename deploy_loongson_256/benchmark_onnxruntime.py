"""Compare OpenCV DNN and ONNX Runtime CPU inference latency."""
from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolo11n-pose-256.onnx"
INPUT_SIZE = 256
INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
RESULT_PATH = ROOT / "benchmark" / "results" / "backend_compare.csv"
CSV_FIELDS = (
    "backend",
    "model_name",
    "input_size",
    "avg_inference_time_ms",
    "fps",
    "speedup_percent",
)
ORT_PROVIDER = "CPUExecutionProvider"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")


def average_forward_ms(forward, blob: np.ndarray, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        forward(blob)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        forward(blob)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(samples_ms))


def benchmark_opencv(blob: np.ndarray, warmup: int, iterations: int) -> float:
    net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    for _ in range(warmup):
        net.setInput(blob)
        net.forward()

    samples_ms = []
    for _ in range(iterations):
        net.setInput(blob)
        start = time.perf_counter()
        net.forward()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(samples_ms))


def load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required for this benchmark. Install a "
            "LoongArch-compatible onnxruntime build on the target system."
        ) from exc
    return ort


def benchmark_onnxruntime(
    ort, blob: np.ndarray, warmup: int, iterations: int
) -> float:
    if ORT_PROVIDER not in ort.get_available_providers():
        raise RuntimeError(
            f"{ORT_PROVIDER} is not available; providers="
            f"{ort.get_available_providers()}"
        )

    session = ort.InferenceSession(
        str(MODEL_PATH), providers=[ORT_PROVIDER]
    )
    if ORT_PROVIDER not in session.get_providers():
        raise RuntimeError(
            f"requested provider {ORT_PROVIDER} was not selected; "
            f"session providers={session.get_providers()}"
        )
    input_meta = session.get_inputs()[0]
    actual_shape = tuple(input_meta.shape)
    if actual_shape != INPUT_SHAPE:
        raise ValueError(
            f"unexpected model input shape: {actual_shape}, expected {INPUT_SHAPE}"
        )

    def forward(input_blob: np.ndarray) -> None:
        session.run(None, {input_meta.name: input_blob})

    return average_forward_ms(forward, blob, warmup, iterations)


def make_row(backend: str, avg_inference_time_ms: float) -> dict[str, str]:
    return {
        "backend": backend,
        "model_name": MODEL_PATH.name,
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "avg_inference_time_ms": f"{avg_inference_time_ms:.3f}",
        "fps": f"{1000.0 / avg_inference_time_ms:.3f}",
        "speedup_percent": "0.000",
    }


def write_results(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")

    ort = load_onnxruntime()

    # One fixed FP32 input keeps preprocessing and allocation outside timing.
    blob = np.zeros(INPUT_SHAPE, dtype=np.float32)
    output_path = RESULT_PATH
    print(f"model: {MODEL_PATH.name}")
    print(f"input: {INPUT_SIZE}x{INPUT_SIZE}, CPU, {platform.machine()}")
    print(f"warmup: {args.warmup}, iterations: {args.iterations}")
    print(f"onnxruntime version: {ort.__version__}")
    print(f"execution provider: {ORT_PROVIDER}")

    rows = []
    opencv_row = make_row(
        "OpenCV DNN",
        benchmark_opencv(blob, args.warmup, args.iterations),
    )
    rows.append(opencv_row)
    print(
        f"{opencv_row['backend']}: "
        f"avg_inference_time_ms={opencv_row['avg_inference_time_ms']}, "
        f"fps={opencv_row['fps']}"
    )

    try:
        ort_row = make_row(
            "ONNX Runtime CPUExecutionProvider",
            benchmark_onnxruntime(ort, blob, args.warmup, args.iterations),
        )
    except RuntimeError as exc:
        write_results(output_path, rows)
        print(f"ERROR: {exc}")
        print(
            "OpenCV result was saved, but the comparison is incomplete. "
            "Install a LoongArch-compatible ONNX Runtime package and rerun."
        )
        return 2

    opencv_time_ms = float(opencv_row["avg_inference_time_ms"])
    ort_time_ms = float(ort_row["avg_inference_time_ms"])
    latency_reduction_percent = (
        (opencv_time_ms - ort_time_ms) / opencv_time_ms * 100.0
    )
    fps_increase_percent = (
        (float(ort_row["fps"]) - float(opencv_row["fps"]))
        / float(opencv_row["fps"])
        * 100.0
    )
    ort_row["speedup_percent"] = f"{latency_reduction_percent:.3f}"
    rows.append(ort_row)
    print(
        f"{ort_row['backend']}: "
        f"avg_inference_time_ms={ort_row['avg_inference_time_ms']}, "
        f"fps={ort_row['fps']}"
    )
    write_results(output_path, rows)
    print(f"latency reduction: {latency_reduction_percent:.3f}%")
    print(f"FPS increase: {fps_increase_percent:.3f}%")
    print(f"saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
