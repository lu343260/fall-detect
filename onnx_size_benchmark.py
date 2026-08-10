"""Benchmark fixed-size YOLO pose ONNX models on the target board.

This is an independent benchmark entry point. It does not modify the
Loongson deployment package or the fall-detection logic.

Examples:
    python onnx_size_benchmark.py --model 320=models/yolo11n-pose-320.onnx \
        --model 640=models/yolo11n-pose-640.onnx --source test.jpg
    python onnx_size_benchmark.py --model 320=yolo11n-pose-320.onnx \
        --camera 0 --warmup 10 --iterations 100

Each model argument is ``SIZE=PATH``. SIZE must match the exported square
input size. The decoder keeps the deployment output contract:
``keypoints.xy`` -> (N, 17, 2), ``keypoints.conf`` -> (N, 17).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

CONF_THRES = 0.3
IOU_THRES = 0.45
NUM_KEYPOINTS = 17
NUM_KEYPOINT_VALUES = NUM_KEYPOINTS * 3


def letterbox(frame: np.ndarray, size: int):
    height, width = frame.shape[:2]
    ratio = min(size / height, size / width)
    resized_size = (round(width * ratio), round(height * ratio))
    pad_x = (size - resized_size[0]) / 2.0
    pad_y = (size - resized_size[1]) / 2.0
    resized = cv2.resize(frame, resized_size, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, ratio, (pad_x, pad_y)


def preprocess(frame: np.ndarray, size: int):
    padded, ratio, padding = letterbox(frame, size)
    blob = padded[..., ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1))[None, ...]
    return blob, ratio, padding


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    area_a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0)
    area_b *= np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return intersection / (area_a + area_b - intersection + 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        overlaps = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= IOU_THRES]
    return np.asarray(keep, dtype=np.int64)


def decode(output, frame_shape, ratio, padding):
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(f"unexpected ONNX output shape: {prediction.shape}")
    prediction = prediction[0].transpose(1, 0)
    expected_channels = 4 + 1 + NUM_KEYPOINT_VALUES
    if prediction.shape[1] != expected_channels:
        raise ValueError(
            f"unexpected output channels: {prediction.shape[1]}, "
            f"expected {expected_channels}"
        )

    candidates = prediction[:, 4] >= CONF_THRES
    if not np.any(candidates):
        return np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32)
    prediction = prediction[candidates]
    scores = prediction[:, 4]
    xywh = prediction[:, :4]
    boxes = np.empty_like(xywh)
    boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2.0
    boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2.0
    keypoints = prediction[:, 5:].reshape(-1, NUM_KEYPOINTS, 3)
    keep = nms(boxes, scores)
    boxes, keypoints = boxes[keep], keypoints[keep]
    pad_x, pad_y = padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / ratio
    keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / ratio
    height, width = frame_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keypoints[:, :, 0] = keypoints[:, :, 0].clip(0, width)
    keypoints[:, :, 1] = keypoints[:, :, 1].clip(0, height)
    return boxes, keypoints


def parse_model(value: str):
    try:
        size_text, path_text = value.split("=", 1)
        size = int(size_text)
        if size <= 0 or not path_text:
            raise ValueError
        return size, Path(path_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "model must use SIZE=PATH, for example 320=yolo11n-pose-320.onnx"
        ) from exc


def get_frame(args):
    if args.source:
        image = cv2.imread(str(args.source))
        if image is None:
            raise RuntimeError(f"cannot read source image: {args.source}")
        return image
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera: {args.camera}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("cannot read a frame from camera")
    return frame


def benchmark_one(model_size, model_path, frame, warmup, iterations):
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    load_start = time.perf_counter()
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_name = net.getUnconnectedOutLayersNames()[0]
    load_ms = (time.perf_counter() - load_start) * 1000

    for _ in range(warmup):
        blob, _, _ = preprocess(frame, model_size)
        net.setInput(blob)
        net.forward(output_name)

    preprocess_ms, inference_ms, postprocess_ms, end_to_end_ms = [], [], [], []
    persons = 0
    xy_shape = conf_shape = None
    raw_output_shape = None
    for _ in range(iterations):
        start = time.perf_counter()
        blob, ratio, padding = preprocess(frame, model_size)
        after_pre = time.perf_counter()
        net.setInput(blob)
        output = net.forward(output_name)
        after_infer = time.perf_counter()
        raw_output_shape = tuple(np.asarray(output).shape)
        boxes, keypoints = decode(output, frame.shape, ratio, padding)
        after_post = time.perf_counter()
        preprocess_ms.append((after_pre - start) * 1000)
        inference_ms.append((after_infer - after_pre) * 1000)
        postprocess_ms.append((after_post - after_infer) * 1000)
        end_to_end_ms.append((after_post - start) * 1000)
        persons += len(boxes)
        xy_shape = keypoints[:, :, :2].shape
        conf_shape = keypoints[:, :, 2].shape

    if xy_shape is None or xy_shape[1:] != (17, 2) or conf_shape[1:] != (17,):
        raise RuntimeError(f"output interface check failed: xy={xy_shape}, conf={conf_shape}")

    def stats(values):
        return np.mean(values), np.median(values), np.percentile(values, 90)

    pre = stats(preprocess_ms)
    inf = stats(inference_ms)
    post = stats(postprocess_ms)
    e2e = stats(end_to_end_ms)
    return {
        "size": model_size,
        "model": str(model_path),
        "load_ms": load_ms,
        "raw_output_shape": raw_output_shape,
        "pre_mean_ms": pre[0], "pre_p50_ms": pre[1], "pre_p90_ms": pre[2],
        "infer_mean_ms": inf[0], "infer_p50_ms": inf[1], "infer_p90_ms": inf[2],
        "post_mean_ms": post[0], "post_p50_ms": post[1], "post_p90_ms": post[2],
        "e2e_mean_ms": e2e[0], "e2e_p50_ms": e2e[1], "e2e_p90_ms": e2e[2],
        "fps_p50": 1000.0 / e2e[1] if e2e[1] > 0 else 0.0,
        "inference_fps_p50": 1000.0 / inf[1] if inf[1] > 0 else 0.0,
        "avg_persons": persons / iterations,
        "xy_shape": xy_shape, "conf_shape": conf_shape,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark fixed-size pose ONNX models")
    parser.add_argument("--model", action="append", required=True, type=parse_model,
                        help="model mapping SIZE=PATH; repeat for multiple sizes")
    parser.add_argument("--source", type=Path, help="one image used for repeatable testing")
    parser.add_argument("--camera", type=int, default=0, help="camera index when --source is omitted")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be >= 0 and iterations must be > 0")
    frame = get_frame(args)
    rows = []
    print(f"frame={frame.shape[1]}x{frame.shape[0]}, warmup={args.warmup}, iterations={args.iterations}")
    for size, path in args.model:
        result = benchmark_one(size, path, frame, args.warmup, args.iterations)
        rows.append(result)
        print(
            f"size={size:>4}  load={result['load_ms']:>8.2f} ms  "
            f"infer p50={result['infer_p50_ms']:>8.2f} ms  "
            f"e2e p50={result['e2e_p50_ms']:>8.2f} ms  "
            f"e2e p90={result['e2e_p90_ms']:>8.2f} ms  "
            f"FPS(e2e)={result['fps_p50']:>6.2f}  "
            f"FPS(infer)={result['inference_fps_p50']:>6.2f}  "
            f"raw={result['raw_output_shape']}  "
            f"xy={result['xy_shape']} conf={result['conf_shape']}"
        )
    if args.csv:
        fields = [key for key, value in rows[0].items() if not isinstance(value, tuple)]
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()
