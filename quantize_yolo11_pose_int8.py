"""Create a calibrated, CPU-friendly INT8 QDQ copy of the 256 pose model.

The source model is never overwritten. Calibration data can be an image, a
directory of images, or a video. If no real data is available, use
``--synthetic-calibration`` only for a smoke test; its accuracy is not a
deployment-quality result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo11n-pose-256.onnx"
DEFAULT_OUTPUT = ROOT / "yolo11n-pose-256-int8.onnx"


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (size - nw) // 2, (size - nh) // 2
    return cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    image = letterbox(image, size)[..., ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in extensions)


class Reader(CalibrationDataReader):
    def __init__(self, input_name: str, input_size: int, paths: list[Path], synthetic: int = 0):
        self.input_name = input_name
        self.input_size = input_size
        self.paths = iter(paths)
        self.synthetic = iter(range(synthetic))

    def get_next(self):
        try:
            path = next(self.paths)
        except StopIteration:
            try:
                next(self.synthetic)
            except StopIteration:
                return None
            return {self.input_name: np.zeros((1, 3, self.input_size, self.input_size), np.float32)}
        image = cv2.imread(str(path))
        if image is None:
            return self.get_next()
        return {self.input_name: preprocess(image, self.input_size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path,
                        help="image, image directory, or a video used for calibration")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--synthetic-calibration", type=int, metavar="N",
                        help="generate N zero-image samples for a smoke test")
    parser.add_argument("--quant-format", choices=("qdq", "qoperator"), default="qdq",
                        help="ONNX quantization representation; qoperator may suit OpenCV DNN better")
    args = parser.parse_args()
    model = args.model if args.model.is_absolute() else ROOT / args.model
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")
    if output.resolve() == model.resolve():
        raise ValueError("output must differ from the original FP32 model")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    meta = session.get_inputs()[0]
    shape = meta.shape
    size = int(shape[-1]) if isinstance(shape[-1], (int, np.integer)) else 256
    paths = image_paths(args.calibration)[:args.max_images] if args.calibration else []
    if args.calibration and args.calibration.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        paths = []
        cap = cv2.VideoCapture(str(args.calibration))
        while len(paths) < args.max_images:
            ok, frame = cap.read()
            if not ok:
                break
            # Keep frames in memory by writing no files: the reader below accepts arrays via closure.
            paths.append(frame)  # type: ignore[arg-type]
        cap.release()
    if not paths and not args.synthetic_calibration:
        raise ValueError("provide --calibration or --synthetic-calibration")

    class ArrayReader(CalibrationDataReader):
        def __init__(self, arrays): self.arrays = iter(arrays)
        def get_next(self):
            try: return {meta.name: preprocess(next(self.arrays), size)}
            except StopIteration: return None

    reader = ArrayReader(paths) if paths and isinstance(paths[0], np.ndarray) else Reader(meta.name, size, paths, args.synthetic_calibration or 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(str(model), str(output), reader,
                    quant_format=QuantFormat.QDQ if args.quant_format == "qdq" else QuantFormat.QOperator,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    # Keep per-tensor scales for opset-12/OpenCV-DNN compatibility.
                    # Per-channel bias quantization can emit DequantizeLinear(axis=...)
                    # which opset 12 runtimes reject.
                    per_channel=False, reduce_range=False, op_types_to_quantize=["Conv", "MatMul"])
    print(f"created: {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
    print("calibration:", "real data" if paths else f"synthetic ({args.synthetic_calibration} samples)")


if __name__ == "__main__":
    main()
