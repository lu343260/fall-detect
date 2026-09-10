"""Loongson deployment inference using OpenCV DNN.

The public interface intentionally matches the current pose_video.py usage:
``model(frame)[0].keypoints.xy`` and ``.conf``.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import time
from urllib import error as urllib_error
from urllib import request as urllib_request
import uuid

import cv2
import numpy as np


CONF_THRES = 0.3
IOU_THRES = 0.45
IMGSZ = 256
NUM_KEYPOINTS = 17
NUM_KEYPOINT_VALUES = NUM_KEYPOINTS * 3

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _available_dnn_options(names):
    return {
        name: getattr(cv2.dnn, constant)
        for name, constant in names.items()
        if hasattr(cv2.dnn, constant)
    }


BACKENDS = _available_dnn_options({
    "opencv": "DNN_BACKEND_OPENCV",
    "halide": "DNN_BACKEND_HALIDE",
    "inference_engine": "DNN_BACKEND_INFERENCE_ENGINE",
    "cuda": "DNN_BACKEND_CUDA",
    "vkcom": "DNN_BACKEND_VKCOM",
})
TARGETS = _available_dnn_options({
    "cpu": "DNN_TARGET_CPU",
    "opencl": "DNN_TARGET_OPENCL",
    "opencl_fp16": "DNN_TARGET_OPENCL_FP16",
    "myriad": "DNN_TARGET_MYRIAD",
    "fpga": "DNN_TARGET_FPGA",
    "cuda": "DNN_TARGET_CUDA",
    "cuda_fp16": "DNN_TARGET_CUDA_FP16",
    "vulkan": "DNN_TARGET_VULKAN",
})


def print_opencv_cpu_info():
    """Print the build options that affect CPU DNN performance."""
    lines = cv2.getBuildInformation().splitlines()
    print("[OpenCV] CPU optimization information:")
    in_cpu_section = False
    printed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("CPU/HW features:", "CPU features:")):
            in_cpu_section = True
            print(f"[OpenCV] {stripped}")
            printed = True
        elif in_cpu_section and stripped:
            print(f"[OpenCV] {stripped}")
        elif in_cpu_section:
            in_cpu_section = False
        if stripped.startswith("Parallel framework:"):
            print(f"[OpenCV] {stripped}")
            printed = True
    if not printed:
        print("[OpenCV] CPU optimization information unavailable")


def configure_dnn(threads: int, backend: str, target: str):
    """Apply process-wide OpenCV threading and DNN execution settings."""
    if threads < 0:
        raise ValueError(f"threads must be >= 0, got {threads}")
    cv2.setNumThreads(threads)
    print(f"[DNN] threads={cv2.getNumThreads()} backend={backend} target={target}")
    print_opencv_cpu_info()


def letterbox(img: np.ndarray, new_shape):
    """Resize with padding and return the image plus scale/padding metadata."""
    height, width = img.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (round(width * ratio), round(height * ratio))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, ratio, (dw, dh)


def preprocess(frame: np.ndarray, imgsz: int):
    padded, ratio, padding = letterbox(frame, (imgsz, imgsz))
    blob = padded[..., ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1))[None, ...]
    return blob, ratio, padding


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between one xyxy box and an array of xyxy boxes."""
    intersection_left = np.maximum(box[0], boxes[:, 0])
    intersection_top = np.maximum(box[1], boxes[:, 1])
    intersection_right = np.minimum(box[2], boxes[:, 2])
    intersection_bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(intersection_right - intersection_left, 0)
    intersection *= np.maximum(intersection_bottom - intersection_top, 0)

    area_a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0)
    area_b *= np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return intersection / (area_a + area_b - intersection + 1e-7)


def nms(boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Greedy class-agnostic NMS; the exported model contains person only."""
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


def scale_boxes_and_keypoints(
    boxes: np.ndarray,
    keypoints: np.ndarray,
    orig_shape,
    ratio: float,
    padding,
    imgsz: int,
):
    """Undo an ``imgsz`` square letterbox and map coordinates to the camera frame."""
    if imgsz <= 0:
        raise ValueError(f"imgsz must be positive, got {imgsz}")
    pad_x, pad_y = padding
    boxes = boxes.copy()
    keypoints = keypoints.copy()

    # Model coordinates are expressed in the square input image. Clamp them
    # before removing letterbox padding so this function stays tied to the
    # detector instance's configured input size.
    boxes[:, :4] = boxes[:, :4].clip(0, imgsz)
    keypoints[:, :, :2] = keypoints[:, :, :2].clip(0, imgsz)

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= ratio
    keypoints[:, :, 0] = (keypoints[:, :, 0] - pad_x) / ratio
    keypoints[:, :, 1] = (keypoints[:, :, 1] - pad_y) / ratio

    height, width = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keypoints[:, :, 0] = keypoints[:, :, 0].clip(0, width)
    keypoints[:, :, 1] = keypoints[:, :, 1].clip(0, height)
    return boxes, keypoints


def postprocess(output, orig_shape, ratio, padding, imgsz: int):
    """Decode the model output and map coordinates back to the camera frame."""
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(f"Unexpected ONNX output shape: {prediction.shape}")
    prediction = prediction[0].transpose(1, 0)

    if prediction.shape[1] != 4 + 1 + NUM_KEYPOINT_VALUES:
        raise ValueError(f"Unexpected ONNX channel count: {prediction.shape[1]}")

    scores = prediction[:, 4]
    candidates = scores >= CONF_THRES
    if not np.any(candidates):
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    prediction = prediction[candidates]
    scores = prediction[:, 4]
    xywh = prediction[:, :4]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    keypoints = prediction[:, 5:].reshape(-1, NUM_KEYPOINTS, 3)

    keep = nms(boxes, scores)
    boxes = boxes[keep]
    keypoints = keypoints[keep]
    scores = scores[keep]
    boxes, keypoints = scale_boxes_and_keypoints(
        boxes, keypoints, orig_shape, ratio, padding, imgsz,
    )
    return boxes, keypoints, scores


class Keypoints:
    def __init__(self, xy: np.ndarray, conf: np.ndarray):
        self.xy = xy
        self.conf = conf


class PoseResult:
    def __init__(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        keypoints: np.ndarray,
        inference_time_ms: float | None = None,
    ):
        self._frame = frame
        self.boxes = boxes
        self.scores = None
        self.inference_time_ms = inference_time_ms
        self.keypoints = Keypoints(
            keypoints[:, :, :2].astype(np.float32),
            keypoints[:, :, 2].astype(np.float32),
        )

    def plot(self) -> np.ndarray:
        image = self._frame.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        xy = self.keypoints.xy
        conf = self.keypoints.conf
        for person_index in range(len(xy)):
            for point_index in range(NUM_KEYPOINTS):
                if conf[person_index, point_index] >= 0.5:
                    x, y = xy[person_index, point_index].astype(int)
                    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)
            for first, second in SKELETON:
                if conf[person_index, first] >= 0.5 and conf[person_index, second] >= 0.5:
                    p1 = tuple(xy[person_index, first].astype(int))
                    p2 = tuple(xy[person_index, second].astype(int))
                    cv2.line(image, p1, p2, (255, 0, 0), 2)
        return image


class RemoteInferenceError(RuntimeError):
    """Raised when the remote inference response cannot be used."""


def _multipart_jpeg(jpeg_bytes: bytes) -> tuple[bytes, str]:
    boundary = f"----loongson-client-{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    body = b"--" + boundary_bytes + b"\r\n"
    body += b'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n'
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += jpeg_bytes + b"\r\n--" + boundary_bytes + b"--\r\n"
    return body, f"multipart/form-data; boundary={boundary}"


def _as_remote_arrays(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    required = ("boxes", "keypoints", "scores", "inference_time_ms")
    missing = [name for name in required if name not in payload]
    if missing:
        raise RemoteInferenceError(
            f"server response is missing fields: {', '.join(missing)}"
        )
    try:
        boxes = np.asarray(payload["boxes"], dtype=np.float32)
        keypoints = np.asarray(payload["keypoints"], dtype=np.float32)
        scores = np.asarray(payload["scores"], dtype=np.float32)
        inference_time_ms = float(payload["inference_time_ms"])
    except (TypeError, ValueError) as exc:
        raise RemoteInferenceError("server response contains invalid numeric data") from exc

    if boxes.size == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
    elif boxes.ndim == 1 and boxes.shape == (4,):
        boxes = boxes.reshape(1, 4)
    if keypoints.size == 0:
        keypoints = np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
    elif keypoints.ndim == 2 and keypoints.shape == (NUM_KEYPOINTS, 3):
        keypoints = keypoints.reshape(1, NUM_KEYPOINTS, 3)
    elif keypoints.ndim == 3 and keypoints.shape[1:] == (NUM_KEYPOINTS, 2):
        try:
            person_scores = scores.reshape(-1, 1, 1)
            keypoints = np.concatenate(
                (keypoints, np.broadcast_to(person_scores, (*keypoints.shape[:2], 1))),
                axis=2,
            )
        except ValueError as exc:
            raise RemoteInferenceError(
                "coordinate-only keypoints do not match scores"
            ) from exc
    if scores.ndim == 0 and scores.size:
        scores = scores.reshape(1)

    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise RemoteInferenceError(f"invalid boxes shape: {boxes.shape}")
    if keypoints.ndim != 3 or keypoints.shape[1:] != (NUM_KEYPOINTS, 3):
        raise RemoteInferenceError(f"invalid keypoints shape: {keypoints.shape}")
    if scores.ndim != 1 or len(boxes) != len(keypoints) or len(scores) != len(boxes):
        raise RemoteInferenceError(
            "server response has inconsistent boxes, keypoints, and scores lengths"
        )
    if not np.isfinite(inference_time_ms) or inference_time_ms < 0:
        raise RemoteInferenceError("invalid inference_time_ms")
    return boxes, keypoints, scores, inference_time_ms


class RemotePoseDetector:
    """Send JPEG frames to the inference service and expose PoseResult output."""

    def __init__(self, server_url: str, timeout: float = 10.0):
        if not server_url:
            raise ValueError("server_url must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.server_url = server_url
        self.timeout = timeout
        self.imgsz = IMGSZ
        self.input_shape = None

    def __call__(self, frame: np.ndarray):
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RemoteInferenceError("failed to encode camera frame as JPEG")
        body, content_type = _multipart_jpeg(encoded.tobytes())
        request = urllib_request.Request(
            self.server_url,
            data=body,
            headers={"Content-Type": content_type, "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                response_body = response.read()
        except urllib_error.HTTPError as exc:
            raise RemoteInferenceError(f"server returned HTTP {exc.code}") from exc
        except (urllib_error.URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            raise RemoteInferenceError(f"request failed: {reason}") from exc
        if status < 200 or status >= 300:
            raise RemoteInferenceError(f"server returned HTTP {status}")
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteInferenceError("server returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RemoteInferenceError("server JSON response must be an object")
        if payload.get("error"):
            raise RemoteInferenceError(f"server error: {payload['error']}")
        boxes, keypoints, scores, inference_time_ms = _as_remote_arrays(payload)
        result = PoseResult(frame, boxes, keypoints, inference_time_ms)
        result.scores = scores
        return [result]


class ONNXPoseDetector:
    def __init__(
        self,
        onnx_path: str = "yolo11n-pose.onnx",
        imgsz: int = IMGSZ,
        threads: int = 0,
        backend: str = "opencv",
        target: str = "cpu",
    ):
        if not isinstance(imgsz, int) or imgsz <= 0:
            raise ValueError(f"imgsz must be a positive integer, got {imgsz!r}")
        if backend not in BACKENDS:
            raise ValueError(f"unsupported DNN backend: {backend}")
        if target not in TARGETS:
            raise ValueError(f"unsupported DNN target: {target}")
        self.imgsz = imgsz
        configure_dnn(threads, backend, target)
        path = Path(onnx_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path.resolve()}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(BACKENDS[backend])
        self.net.setPreferableTarget(TARGETS[target])
        self.output_name = self.net.getUnconnectedOutLayersNames()[0]

        # OpenCV DNN remains the actual inference backend on LoongArch. The
        # blob shape is the concrete shape passed to the ONNX network. ONNX
        # Runtime is optional and is used only for metadata verification when
        # it happens to be installed; it is never required for deployment.
        self.input_shape = (1, 3, self.imgsz, self.imgsz)
        self.input_name = None
        try:
            import onnxruntime as ort
        except ImportError:
            pass
        else:
            metadata_session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            input_meta = metadata_session.get_inputs()[0]
            self.input_name = input_meta.name
            actual_shape = tuple(input_meta.shape)
            if actual_shape != self.input_shape:
                raise ValueError(
                    f"ONNX input shape mismatch: model={actual_shape}, "
                    f"configured={self.input_shape}"
                )
        print(
            f"[MODEL] path={path.name} backend=OpenCV-DNN "
            f"input_shape={self.input_shape}"
        )
        if self.input_name is not None:
            print(f"[MODEL] optional ONNX metadata input_name={self.input_name}")

    def __call__(self, frame: np.ndarray):
        blob, ratio, padding = preprocess(frame, self.imgsz)
        self.net.setInput(blob)
        start = time.time()
        output = self.net.forward(self.output_name)
        print(f"[DNN] forward time={(time.time() - start) * 1000:.2f} ms")
        boxes, keypoints, scores = postprocess(
            output, frame.shape[:2], ratio, padding, self.imgsz,
        )
        result = PoseResult(frame, boxes, keypoints)
        result.scores = scores
        return [result]
