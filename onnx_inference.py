"""
onnx_inference.py —— ONNX Runtime 姿态推理封装
用于替换 pose_video.py 中的 YOLO(.pt) 推理，接口保持兼容：

    model = ONNXPoseDetector("yolo11n-pose.onnx")
    results = model(frame)                    # 返回 [PoseResult]，兼容 results[0]
    keypoints_xy   = results[0].keypoints.xy     # (N,17,2)
    keypoints_conf = results[0].keypoints.conf   # (N,17)
    annotated_frame = results[0].plot()          # 基础骨架绘制

预处理/后处理与 onnx_test.py 已验证的管线一致：
letterbox -> BGR转RGB -> /255 -> onnxruntime 推理 -> NMS + scale_coords 还原到原图。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

CONF_THRES = 0.3    # 与 pose_video.py 原 conf=0.3 一致
IOU_THRES = 0.45    # ultralytics 默认 NMS IoU
IMGSZ = 640         # 与 export_onnx.py / onnx_test.py 一致
NC = 1              # pose 模型只有 person 一个类别

# COCO-17 骨架连线（用于 plot 绘制）
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 面部
    (5, 6),                                  # 肩-肩
    (5, 7), (7, 9),                          # 左臂
    (6, 8), (8, 10),                         # 右臂
    (5, 11), (6, 12), (11, 12),              # 躯干
    (11, 13), (13, 15),                      # 左腿
    (12, 14), (14, 16),                      # 右腿
]


def letterbox(img: np.ndarray, new_shape=(IMGSZ, IMGSZ), center=True, scaleup=True) -> np.ndarray:
    """与 ultralytics LetterBox 保持一致的缩放 + 居中填充（padding=114）。"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (round(shape[1] * r), round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if center:
        dw /= 2.0
        dh /= 2.0
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded


def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR 帧 -> letterbox 640x640 -> BGR转RGB -> /255 -> (1,3,640,640) float32。"""
    lb = letterbox(frame)
    blob = lb[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1))[None, ...]
    return blob


def postprocess(output: np.ndarray, orig_shape):
    """ONNX 输出 (1,56,8400) -> 检测框 + 17 关键点(已还原到原图)。

    56 通道 = 4(框 cxcywh) + 1(类别置信度 person) + 51(17点 x,y,conf)。
    后处理复用 ultralytics 的 non_max_suppression / scale_coords，
    与 .pt 推理路径完全一致，保证关键点精度一致。
    """
    from ultralytics.utils import nms, ops

    pred = torch.from_numpy(output)  # (1, 56, 8400)
    det = nms.non_max_suppression(
        pred,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        classes=[0],
        max_det=300,
        nc=NC,
    )[0]
    if det.numel() == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 17, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    boxes = ops.scale_boxes((IMGSZ, IMGSZ), det[:, :4].clone(), orig_shape)
    kpts = ops.scale_coords((IMGSZ, IMGSZ), det[:, 6:].view(-1, 17, 3).clone(), orig_shape)
    confs = det[:, 4].cpu().numpy()
    return boxes.cpu().numpy(), kpts.cpu().numpy(), confs


class Keypoints:
    """模拟 results[0].keypoints：xy (N,17,2)、conf (N,17)。"""

    def __init__(self, xy: np.ndarray, conf: np.ndarray):
        self.xy = xy    # (N,17,2) float32
        self.conf = conf  # (N,17) float32


class PoseResult:
    """模拟单个检测结果对象，兼容 results[0].keypoints / results[0].plot()。"""

    def __init__(self, frame: np.ndarray, boxes: np.ndarray, kpts: np.ndarray):
        self._frame = frame  # 缓存原图，供 plot() 使用
        self.boxes = boxes   # (N,4) xyxy
        if len(kpts) > 0:
            self.keypoints = Keypoints(kpts[:, :, :2].astype(np.float32),
                                       kpts[:, :, 2].astype(np.float32))
        else:
            self.keypoints = Keypoints(np.zeros((0, 17, 2), dtype=np.float32),
                                       np.zeros((0, 17), dtype=np.float32))

    def plot(self) -> np.ndarray:
        """基础骨架绘制：检测框 + 关键点圆点 + 骨架连线（保留原 plot 功能）。"""
        img = self._frame.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        xy = self.keypoints.xy
        conf = self.keypoints.conf
        for i in range(len(xy)):
            # 关键点圆点
            for j in range(17):
                if conf[i, j] >= 0.5:
                    x, y = int(xy[i, j, 0]), int(xy[i, j, 1])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            # 骨架连线
            for a, b in SKELETON:
                if conf[i, a] >= 0.5 and conf[i, b] >= 0.5:
                    p1 = (int(xy[i, a, 0]), int(xy[i, a, 1]))
                    p2 = (int(xy[i, b, 0]), int(xy[i, b, 1]))
                    cv2.line(img, p1, p2, (255, 0, 0), 2)
        return img


class ONNXPoseDetector:
    """ONNX Runtime 姿态检测器，接口兼容 ultralytics YOLO 的 model(frame) 调用。"""

    def __init__(self, onnx_path: str = "yolo11n-pose.onnx",
                 conf: float = CONF_THRES, iou: float = IOU_THRES, imgsz: int = IMGSZ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        path = Path(onnx_path)
        if not path.is_absolute():
            # 相对路径基于本文件所在目录解析，保证从任何目录运行都能找到模型
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(f"未找到 ONNX 模型: {path.resolve()}")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, frame: np.ndarray):
        """输入 BGR 帧，返回 [PoseResult]，兼容 results[0] 用法。"""
        blob = preprocess(frame)
        out = self.session.run(self.output_names, {self.input_name: blob})[0]
        boxes, kpts, _ = postprocess(out, frame.shape[:2])
        return [PoseResult(frame, boxes, kpts)]