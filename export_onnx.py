"""
导出 YOLO11n-pose 为 ONNX 格式。
用法: python export_onnx.py
输出: 项目目录下的 yolo11n-pose.onnx
不会修改项目中的其他文件。
"""
from pathlib import Path

from ultralytics import YOLO

# 脚本所在目录 = 项目根目录
ROOT = Path(__file__).resolve().parent
PT_PATH = ROOT / "yolo11n-pose.pt"
ONNX_PATH = ROOT / "yolo11n-pose.onnx"


def main():
    if not PT_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件: {PT_PATH}")

    print(f"[1/3] 加载模型: {PT_PATH.name}")
    model = YOLO(str(PT_PATH))

    print("[2/3] 导出 ONNX (imgsz=640, opset=12) ...")
    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        dynamic=False,
        simplify=True,
    )
    print(f"[3/3] 导出完成: {exported}")

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"导出失败，未生成: {ONNX_PATH}")
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"ONNX 文件: {ONNX_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
