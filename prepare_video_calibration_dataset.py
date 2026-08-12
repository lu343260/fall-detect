"""Extract non-adjacent calibration and evaluation frames from camera videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def extract(video: Path, output: Path, count: int, prefix: str) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"cannot read frames: {video}")
    output.mkdir(parents=True, exist_ok=True)
    # Evenly spaced frame indices avoid adjacent-frame duplication and cover the video.
    indices = [round(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
    rows = []
    for number, index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        path = output / f"{prefix}_{number:04d}_frame{index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"cannot write {path}")
        rows.append({"file": str(path), "video": str(video), "frame": index,
                     "time_s": index / (cap.get(cv2.CAP_PROP_FPS) or 1.0)})
    cap.release()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-video", type=Path, required=True)
    parser.add_argument("--calibration-per-video", type=int, default=65)
    parser.add_argument("--evaluation-frames", type=int, default=60)
    args = parser.parse_args()
    videos = sorted(args.source_dir.glob("*.mp4"))
    evaluation = args.evaluation_video.resolve()
    if len(videos) < 2:
        raise ValueError("need at least two videos for a video-level split")
    if evaluation not in {p.resolve() for p in videos}:
        raise FileNotFoundError(f"evaluation video is not in source directory: {evaluation}")
    calibration_videos = [p for p in videos if p.resolve() != evaluation]
    calibration_rows = []
    for video in calibration_videos:
        calibration_rows += extract(video, args.output_dir / "calibration", args.calibration_per_video, video.stem)
    evaluation_rows = extract(evaluation, args.output_dir / "evaluation", args.evaluation_frames, evaluation.stem)
    metadata = {"calibration_videos": [str(p) for p in calibration_videos],
                "evaluation_video": str(evaluation), "calibration": calibration_rows,
                "evaluation": evaluation_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"calibration_frames={len(calibration_rows)}")
    print(f"evaluation_frames={len(evaluation_rows)}")
    print(f"metadata={args.output_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
