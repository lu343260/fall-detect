"""Compare FP32/INT8 model size, CPU latency, FPS and keypoint consistency."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
KEYPOINTS = {"nose": 0, "left_shoulder": 5, "right_shoulder": 6,
             "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14}


def preprocess(frame, size):
    h, w = frame.shape[:2]; scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    image = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, (size - nh) // 2, size - nh - (size - nh) // 2,
                               (size - nw) // 2, size - nw - (size - nw) // 2,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray((image[..., ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None])


def detections(output):
    pred = np.asarray(output, dtype=np.float32)[0].T
    pred = pred[pred[:, 4] >= 0.3]
    if not len(pred): return np.empty((0, 4), np.float32), np.empty((0, 17, 3), np.float32)
    boxes = np.c_[pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2]
    order = np.argsort(pred[:, 4])[::-1]
    return boxes[order], pred[order, 5:].reshape(-1, 17, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp32", type=Path, default=ROOT / "yolo11n-pose-256.onnx")
    p.add_argument("--int8", type=Path, default=ROOT / "yolo11n-pose-256-int8.onnx")
    p.add_argument("--source", type=Path, required=True, help="evaluation image or directory")
    p.add_argument("--warmup", type=int, default=10); p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--report", type=Path, default=ROOT / "int8_comparison_report.json")
    args = p.parse_args()
    sources = [args.source] if args.source.is_file() else sorted(args.source.glob("*.jpg"))
    if not sources: raise RuntimeError(f"no evaluation images found: {args.source}")
    rows = []
    outputs = {}
    for label, path in (("FP32", args.fp32), ("INT8", args.int8)):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]; size = int(inp.shape[-1]); names = [o.name for o in session.get_outputs()]
        first = cv2.imread(str(sources[0]))
        if first is None: raise RuntimeError(f"cannot read source image: {sources[0]}")
        blob = preprocess(first, size)
        for _ in range(args.warmup): session.run(names, {inp.name: blob})
        times = []
        output_pairs = []
        for source in sources:
            frame = cv2.imread(str(source))
            if frame is None: continue
            current = preprocess(frame, size)
            repeat = args.iterations if len(sources) == 1 else 1
            for _ in range(repeat):
                start = time.perf_counter(); out = session.run(names, {inp.name: current})[0]
                times.append((time.perf_counter() - start) * 1000)
            output_pairs.append(detections(out))
        outputs[label] = output_pairs
        mean = float(np.mean(times))
        rows.append({"model": label, "path": str(path), "size_bytes": path.stat().st_size,
                     "size_mib": path.stat().st_size / 1024 / 1024,
                     "inference_mean_ms": mean, "inference_fps": 1000 / mean})
    fp_outputs = outputs["FP32"]; int_outputs = outputs["INT8"]
    consistency = {}
    per_point = {name: {"coordinate_error_px": [], "confidence_abs_delta": []} for name in KEYPOINTS}
    paired = 0
    for (fp_boxes, fp_kpts), (int_boxes, int_kpts) in zip(fp_outputs, int_outputs):
        if len(fp_kpts) and len(int_kpts):
            paired += 1
            errors = np.linalg.norm(fp_kpts[0, :, :2] - int_kpts[0, :, :2], axis=1)
            for name, idx in KEYPOINTS.items():
                per_point[name]["coordinate_error_px"].append(float(errors[idx]))
                per_point[name]["confidence_abs_delta"].append(float(abs(fp_kpts[0, idx, 2] - int_kpts[0, idx, 2])))
    for name, values in per_point.items():
        consistency[name] = {"images_with_both_detections": len(values["coordinate_error_px"]),
                             "coordinate_error_px": float(np.mean(values["coordinate_error_px"])) if values["coordinate_error_px"] else None,
                             "confidence_abs_delta": float(np.mean(values["confidence_abs_delta"])) if values["confidence_abs_delta"] else None}
    report = {"evaluation_images": len(sources), "paired_top_detections": paired, "models": rows,
              "keypoint_consistency_top_detection": consistency,
              "note": "This is output consistency on one source image, not mAP. Use representative labelled fall data for final accuracy."}
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(f"report written: {args.report}")


if __name__ == "__main__": main()
