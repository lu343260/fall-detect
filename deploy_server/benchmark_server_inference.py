"""Benchmark server-side ONNX Runtime inference for YOLO11n-Pose 256x256.

The timed section contains only ``session.run()`` on an already-created FP32
input tensor. Model loading, input creation, preprocessing, postprocessing,
and warmup are outside the timing, matching the existing Loongson forward
benchmarks as closely as possible.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
LOONGSON_INFERENCE_DIR = PROJECT_ROOT / "deploy_loongson_256"
DEFAULT_MODEL = PROJECT_ROOT / "yolo11n-pose-256.onnx"
INPUT_SIZE = 256
INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
DEFAULT_PROVIDER = "CPUExecutionProvider"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=(
            "ONNX Runtime execution provider to request; defaults to CPU and "
            "never auto-selects a GPU provider"
        ),
    )
    parser.add_argument(
        "--intra-op-num-threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads; 0 lets ONNX Runtime choose",
    )
    return parser.parse_args()


def resolve_model(path: Path) -> Path:
    if path.is_absolute():
        resolved = path
    elif path.exists():
        resolved = path.resolve()
    else:
        resolved = (PROJECT_ROOT / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"model not found: {resolved}")
    return resolved


def load_reused_pipeline():
    """Load the existing 256 deployment preprocessing/postprocessing helpers."""
    if str(LOONGSON_INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(LOONGSON_INFERENCE_DIR))
    from onnx_inference import postprocess, preprocess

    return preprocess, postprocess


def summarize(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "average_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def benchmark(session, input_name: str, blob: np.ndarray, warmup: int, iterations: int):
    def forward() -> None:
        session.run(None, {input_name: blob})

    for _ in range(warmup):
        forward()

    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        forward()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    stats = summarize(samples_ms)
    stats["theoretical_fps"] = 1000.0 / stats["average_ms"]
    return stats


def print_system_info(ort, selected_provider: str) -> None:
    print(f"CPU: {platform.processor() or 'unknown'}")
    print(f"CPU architecture: {platform.machine() or 'unknown'}")
    print(f"CPU logical count: {os.cpu_count() or 'unknown'}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Requested provider: {selected_provider}")


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    if args.intra_op_num_threads < 0:
        raise ValueError("intra-op-num-threads must be >= 0")


def main() -> int:
    args = parse_args()
    validate_args(args)
    model_path = resolve_model(args.model)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is required. Install the server requirements first."
        ) from exc

    available_providers = ort.get_available_providers()
    print_system_info(ort, args.provider)
    if args.provider not in available_providers:
        raise RuntimeError(
            f"provider {args.provider!r} is unavailable; "
            f"available providers={available_providers}"
        )

    session_options = ort.SessionOptions()
    if args.intra_op_num_threads:
        session_options.intra_op_num_threads = args.intra_op_num_threads
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[args.provider],
    )
    session_providers = session.get_providers()
    if not session_providers or session_providers[0] != args.provider:
        raise RuntimeError(
            f"requested provider was not selected first; "
            f"session providers={session_providers}"
        )
    print(f"Model load: OK ({model_path})")

    input_meta = session.get_inputs()[0]
    actual_shape = tuple(input_meta.shape)
    if actual_shape != INPUT_SHAPE:
        raise ValueError(
            f"unexpected model input shape: {actual_shape}, expected {INPUT_SHAPE}"
        )

    preprocess, postprocess = load_reused_pipeline()
    frame = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    blob, ratio, padding = preprocess(frame, INPUT_SIZE)
    output = session.run(None, {input_meta.name: blob})[0]
    boxes, keypoints, scores = postprocess(
        output, frame.shape[:2], ratio, padding, INPUT_SIZE
    )

    print(f"Model: {model_path.name}")
    print(f"Input: {actual_shape}, dtype={blob.dtype}")
    print(f"Session providers: {session_providers}")
    print(
        "Pipeline smoke test: "
        f"output_shape={np.asarray(output).shape}, persons={len(boxes)}, "
        f"keypoints_shape={keypoints.shape}, scores_shape={scores.shape}"
    )
    print(f"Warmup: {args.warmup}, iterations: {args.iterations}")
    print("Timing scope: ONNX Runtime session.run() only")

    stats = benchmark(session, input_meta.name, blob, args.warmup, args.iterations)
    print(
        "Inference: "
        f"average={stats['average_ms']:.3f} ms, "
        f"min={stats['min_ms']:.3f} ms, max={stats['max_ms']:.3f} ms, "
        f"P50={stats['p50_ms']:.3f} ms, P95={stats['p95_ms']:.3f} ms"
    )
    print(f"Theoretical FPS: {stats['theoretical_fps']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
