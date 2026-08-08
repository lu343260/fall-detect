"""
onnx_test.py —— ONNX 模型与 .pt 模型关键点输出对比测试（新增文件，不改动已有文件）

功能:
1. 加载 yolo11n-pose.onnx，用 ONNX Runtime 在 CPU 上推理
2. 对测试视频逐帧执行: letterbox 预处理 -> onnxruntime 推理 -> 解析 17 个关键点
3. 输出与当前 YOLO API 兼容的关键点格式 keypoints_xy
4. 与 yolo11n-pose.pt 的结果逐帧对比，统计 hip(11, 12) 坐标误差

用法:
    python onnx_test.py [视频路径 ...] [--max-frames N] [--every N]

示例:
    python onnx_test.py                          # 默认 test_person.mp4
    python onnx_test.py test_person2.mp4
    python onnx_test.py test_person.mp4 test_person2.mp4 --max-frames 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "yolo11n-pose.onnx"
PT_PATH = ROOT / "yolo11n-pose.pt"

CONF_THRES = 0.3   # 与 pose_video.py 中的 conf=0.3 一致
IOU_THRES = 0.45   # ultralytics 默认 NMS IoU
IMGSZ = 640        # 与 export_onnx.py 的 imgsz 一致

# COCO-17 关键点名称（索引即顺序，与 YOLO keypoints API 一致）
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
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


def load_onnx(path: Path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names


def postprocess_onnx(output: np.ndarray, orig_shape):
    """ONNX 输出 (1,56,8400) -> 检测框 + 17 关键点(已还原到原图)。

    56 通道 = 4(框 cxcywh) + 1(类别置信度 person) + 51(17点 x,y,conf)。
    后处理复用 ultralytics 的 non_max_suppression / scale_coords，
    保证与 .pt 推理路径完全一致，对比结果只反映模型本身差异。
    """
    from ultralytics.utils import nms, ops

    pred = torch.from_numpy(output)  # (1, 56, 8400)
    det = nms.non_max_suppression(
        pred,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        classes=[0],
        max_det=300,
        nc=1,
    )[0]
    if det.numel() == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 17, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    boxes = ops.scale_boxes((IMGSZ, IMGSZ), det[:, :4].clone(), orig_shape)
    kpts = ops.scale_coords((IMGSZ, IMGSZ), det[:, 6:].view(-1, 17, 3).clone(), orig_shape)
    confs = det[:, 4].cpu().numpy()
    return boxes.cpu().numpy(), kpts.cpu().numpy(), confs


def run_pt(model, frame: np.ndarray):
    """走当前项目使用的 .pt 推理路径，返回 boxes / keypoints_xy / keypoints_conf。"""
    results = model(frame, conf=CONF_THRES, classes=[0], verbose=False)
    r = results[0]
    kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 2))
    confs = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else np.zeros(len(kpts))
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    return boxes, kpts, confs


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (M,4) b:(N,4) xyxy -> IoU (M,N)。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def match_detections(onnx_boxes: np.ndarray, pt_boxes: np.ndarray):
    """贪心匹配: ONNX 框按置信度从高到低，找 IoU 最大的 .pt 框 (IoU >= 0.3)。"""
    iou = iou_matrix(onnx_boxes, pt_boxes)
    matched = {}
    used = set()
    for i in range(len(onnx_boxes)):
        best_j, best_v = -1, 0.3
        for j in range(len(pt_boxes)):
            if j in used:
                continue
            if iou[i, j] > best_v:
                best_v, best_j = iou[i, j], j
        if best_j >= 0:
            matched[i] = best_j
            used.add(best_j)
    return matched


def format_keypoints(kpts_xy: np.ndarray) -> str:
    """按 YOLO API 风格输出 keypoints_xy（带名称，共 17 点）。"""
    items = []
    for idx, name in enumerate(KPT_NAMES):
        x, y = kpts_xy[idx]
        items.append(f"{name}({x:.1f},{y:.1f})")
    return "[" + ", ".join(items) + "]"


def run_video(model, session, input_name, output_names, video_path: Path,
              max_frames: int = 0, every: int = 1, show_frames: int = 3):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 无法打开视频: {video_path}")
        return None

    frame_idx = 0
    compared = 0
    skipped = 0
    hip_errors = []          # 每个已匹配目标的左/右髋误差
    onnx_times, pt_times = [], []
    shown = 0
    first_samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break
        if frame_idx % every != 0:
            frame_idx += 1
            continue

        orig_shape = frame.shape[:2]

        # --- ONNX 推理 ---
        blob = preprocess(frame)
        t0 = time.perf_counter()
        out = session.run(output_names, {input_name: blob})[0]
        onnx_times.append(time.perf_counter() - t0)
        onnx_boxes, onnx_kpts, onnx_confs = postprocess_onnx(out, orig_shape)

        # --- .pt 推理 ---
        t0 = time.perf_counter()
        pt_boxes, pt_kpts, pt_confs = run_pt(model, frame)
        pt_times.append(time.perf_counter() - t0)

        if len(onnx_boxes) == 0 or len(pt_boxes) == 0:
            skipped += 1
            frame_idx += 1
            continue

        # 按置信度排序 ONNX 检测，再与 .pt 匹配
        onnx_order = np.argsort(-onnx_confs)
        onnx_boxes_s = onnx_boxes[onnx_order]
        onnx_kpts_s = onnx_kpts[onnx_order]
        matched = match_detections(onnx_boxes_s, pt_boxes)
        if not matched:
            skipped += 1
            frame_idx += 1
            continue

        compared += 1
        for i, j in matched.items():
            k_onnx = onnx_kpts_s[i]      # (17,3)
            k_pt = pt_kpts[j]            # (17,2)
            for hip_idx in (11, 12):
                if k_onnx[hip_idx, 2] >= 0.3:
                    dist = float(np.hypot(k_onnx[hip_idx, 0] - k_pt[hip_idx, 0],
                                          k_onnx[hip_idx, 1] - k_pt[hip_idx, 1]))
                    hip_errors.append(dist)

        if shown < show_frames:
            shown += 1
            i = next(iter(matched))
            j = matched[i]
            sample = {
                "frame": frame_idx,
                "onnx": onnx_kpts_s[i][:, :2],
                "pt": pt_kpts[j][:, :2],
                "hip_onnx": onnx_kpts_s[i][11:13, :2],
                "hip_pt": pt_kpts[j][11:13, :2],
            }
            first_samples.append(sample)

        frame_idx += 1

    cap.release()
    return {
        "video": video_path.name,
        "compared": compared,
        "skipped": skipped,
        "hip_errors": hip_errors,
        "onnx_avg_ms": float(np.mean(onnx_times)) * 1000 if onnx_times else 0.0,
        "pt_avg_ms": float(np.mean(pt_times)) * 1000 if pt_times else 0.0,
        "samples": first_samples,
    }


def print_sample(video_name: str, sample):
    print(f"\n{video_name} 帧 {sample['frame']} 关键点对比:")
    print(f"  ONNX keypoints_xy: {format_keypoints(sample['onnx'])}")
    print(f"  PT   keypoints_xy: {format_keypoints(sample['pt'])}")
    print(f"  hip 误差:  left={np.hypot(*(sample['hip_onnx'][0] - sample['hip_pt'][0])):.2f}px  "
          f"right={np.hypot(*(sample['hip_onnx'][1] - sample['hip_pt'][1])):.2f}px")


def main():
    parser = argparse.ArgumentParser(description="ONNX 与 .pt 关键点对比测试")
    parser.add_argument("videos", nargs="*", help="视频路径，默认 test_person.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="每个视频最多处理帧数 (0=全部)")
    parser.add_argument("--every", type=int, default=1, help="每隔 N 帧处理一帧")
    args = parser.parse_args()

    videos = [Path(v) for v in args.videos] or [ROOT / "test_person.mp4"]
    for v in videos:
        if not v.is_absolute():
            v = ROOT / v
        v = v.resolve()

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"未找到 ONNX 模型: {ONNX_PATH}，请先运行 export_onnx.py")
    if not PT_PATH.exists():
        raise FileNotFoundError(f"未找到 .pt 模型: {PT_PATH}")

    print(f"[1/3] 加载 ONNX: {ONNX_PATH.name}")
    session, input_name, output_names = load_onnx(ONNX_PATH)
    print(f"      输入: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"      输出: {[f'{o.name} {o.shape}' for o in session.get_outputs()]}")

    print(f"[2/3] 加载 .pt 模型: {PT_PATH.name} (用于对比)")
    from ultralytics import YOLO
    model = YOLO(str(PT_PATH))

    print("[3/3] 开始逐视频对比 ...\n")
    all_hip_errors = []
    ok_videos = []
    for v in videos:
        if not v.exists():
            print(f"[跳过] 文件不存在: {v}")
            continue
        stats = run_video(model, session, input_name, output_names, v,
                          max_frames=args.max_frames, every=args.every)
        if stats is None:
            continue
        for s in stats["samples"]:
            print_sample(stats["video"], s)
        errors = stats["hip_errors"]
        all_hip_errors.extend(errors)
        mean_e = float(np.mean(errors)) if errors else float("nan")
        max_e = float(np.max(errors)) if errors else float("nan")
        over5 = int(np.sum(np.array(errors) > 5.0)) if errors else 0
        p95 = float(np.percentile(errors, 95)) if errors else float("nan")
        p99 = float(np.percentile(errors, 99)) if errors else float("nan")
        ratio = (over5 / len(errors) * 100) if errors else 0.0
        # 通过标准: 均值 <=5px 且 95% 样本 <=5px（高速跌倒瞬间允许个别帧略超）
        verdict = "通过" if errors and mean_e <= 5.0 and p95 <= 5.0 else ("无有效对比" if not errors else "未通过")
        if verdict == "通过":
            ok_videos.append(stats["video"])
        print(f"\n===== {stats['video']} =====")
        print(f"  有效对比帧: {stats['compared']}  无检测跳过帧: {stats['skipped']}")
        print(f"  hip(11,12) 误差: 均值 {mean_e:.2f}px | p95 {p95:.2f}px | p99 {p99:.2f}px | 最大 {max_e:.2f}px")
        print(f"  超 5px 样本: {over5}/{len(errors)} ({ratio:.1f}%)")
        print(f"  推理耗时: ONNX {stats['onnx_avg_ms']:.1f}ms/帧 | PT {stats['pt_avg_ms']:.1f}ms/帧")
        print(f"  结论: 均值与 p95 均<=5px -> {verdict}")

    if all_hip_errors:
        arr = np.array(all_hip_errors)
        print(f"\n========== 汇总 ==========")
        print(f"  全部 hip 样本: {len(arr)} 个 | 均值 {np.mean(arr):.2f}px | p95 {np.percentile(arr,95):.2f}px | "
              f"最大 {np.max(arr):.2f}px | >5px {int(np.sum(arr > 5.0))} 个 ({np.sum(arr > 5.0)/len(arr)*100:.1f}%)")
        print(f"  通过视频: {ok_videos if ok_videos else '无'}")
    else:
        print("\n没有可对比的 hip 数据（所有帧均无检测或匹配失败）。")


if __name__ == "__main__":
    main()