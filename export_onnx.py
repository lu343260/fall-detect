"""Export fixed-size YOLO11n-pose ONNX models.

The default command exports 320, 256, and 192 models with independent
filenames. It never targets the existing 640 model.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PT_PATH = ROOT / "yolo11n-pose.pt"
DEFAULT_SIZES = (320, 256, 192)


def export_one(pt_path: Path, output_dir: Path, size: int) -> Path:
    """Export one size while avoiding Ultralytics' stem-based name collision."""
    output_path = output_dir / f"yolo11n-pose-{size}.onnx"
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing model: {output_path}. "
            "Move it aside or remove it explicitly before exporting."
        )

    with tempfile.TemporaryDirectory(prefix=f"yolo11n-pose-{size}-") as temp_dir:
        temp_pt = Path(temp_dir) / f"yolo11n-pose-{size}.pt"
        shutil.copy2(pt_path, temp_pt)
        model = YOLO(str(temp_pt))
        exported = model.export(
            format="onnx",
            imgsz=size,
            opset=12,
            dynamic=False,
            simplify=True,
        )
        exported_path = Path(exported)
        if not exported_path.exists():
            raise FileNotFoundError(f"export failed, output not found: {exported_path}")
        shutil.copy2(exported_path, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="square input sizes (default: 320 256 192)")
    parser.add_argument("--pt", type=Path, default=PT_PATH,
                        help="source YOLO pose .pt model")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="directory for independent ONNX files")
    args = parser.parse_args()

    pt_path = args.pt if args.pt.is_absolute() else ROOT / args.pt
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not pt_path.exists():
        raise FileNotFoundError(f"model not found: {pt_path}")
    if any(size <= 0 for size in args.sizes):
        parser.error("all sizes must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source model: {pt_path}")
    for size in args.sizes:
        print(f"exporting imgsz={size} ...")
        output_path = export_one(pt_path, output_dir, size)
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"created: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
