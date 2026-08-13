"""Run the OpenCV DNN benchmark for a set of thread configurations."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import cv2

import benchmark_opencv_forward as benchmark_module


ROOT = Path(__file__).resolve().parent
DEFAULT_THREADS = (0, 1, 2, 4, 8, 12)
RESULTS_DIR = ROOT / "benchmark" / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_result.csv"
CSV_FIELDS = (
    "model_name",
    "input_size",
    "thread_num",
    "avg_latency",
    "min_latency",
    "max_latency",
    "fps",
    "timestamp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=benchmark_module.DEFAULT_MODEL)
    parser.add_argument(
        "--input-size", type=int, default=benchmark_module.DEFAULT_INPUT_SIZE
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="number of benchmark runs for each thread configuration",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=list(DEFAULT_THREADS),
        help="thread configurations to test (default: 0 1 2 4 8 12)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_FILE,
        help=f"CSV output path (default: {RESULTS_FILE})",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.input_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("input-size must be > 0, warmup >= 0, iterations > 0")
    if args.runs <= 0:
        raise ValueError("runs must be > 0")
    if any(thread < 0 for thread in args.threads):
        raise ValueError("threads must be >= 0")


def _benchmark_worker(
    model_path: str,
    input_size: int,
    warmup: int,
    iterations: int,
    thread_num: int,
    result_queue,
) -> None:
    if thread_num != 0:
        cv2.setNumThreads(thread_num)
    result = benchmark_module.benchmark(
        Path(model_path), input_size, warmup, iterations
    )
    result_queue.put(result)


def run_one(model_path: Path, args: argparse.Namespace, thread_num: int) -> dict:
    # A fresh process keeps --threads 0 isolated from earlier thread settings.
    context = get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_benchmark_worker,
        args=(
            str(model_path),
            args.input_size,
            args.warmup,
            args.iterations,
            thread_num,
            result_queue,
        ),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"benchmark failed for threads={thread_num} "
            f"(exit code {process.exitcode})"
        )
    result = result_queue.get()
    total_stats = result["setInput_plus_forward"]
    return {
        "model_name": result["model_name"],
        "input_size": result["input_size"],
        "thread_num": thread_num,
        "avg_latency": f"{total_stats['average_ms']:.3f}",
        "min_latency": f"{total_stats['min_ms']:.3f}",
        "max_latency": f"{total_stats['max_ms']:.3f}",
        "fps": f"{result['theoretical_fps']:.3f}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    model_path = benchmark_module.resolve_model(args.model)
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for thread_num in args.threads:
        for run_number in range(1, args.runs + 1):
            print(
                f"Running threads={thread_num}, run={run_number}/{args.runs}"
            )
            row = run_one(model_path, args, thread_num)
            rows.append(row)
            print(
                f"  average={row['avg_latency']} ms, FPS={row['fps']}"
            )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark results to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
