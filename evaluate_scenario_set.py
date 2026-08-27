"""Run the scenario regression set without changing the deployment entry point.

The evaluator reuses ``onnx_inference.ONNXPoseDetector`` and
``fall_detector.FallDetector``.  It writes reports under the evaluation dataset
and never changes model files or detector thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_POINTS = (5, 6, 11, 12)
TRUE_VALUES = {"1", "true", "yes", "y", "alarm", "fall", "fallen"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate datasets/fall_eval_v1 manifest with the existing ONNX pose pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/fall_eval_v1"),
        help="Scenario dataset directory (default: datasets/fall_eval_v1).",
    )
    parser.add_argument("--model", type=Path, required=True, help="ONNX model path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report directory. Defaults to <dataset>/reports/<run_id>.",
    )
    parser.add_argument(
        "--alarm-match-window",
        type=float,
        default=5.0,
        help="Seconds after fall_start_s in which an alarm matches a fall (default: 5).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Keypoint confidence used for missing-keypoint statistics.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional per-video frame limit for a smoke test.",
    )
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def first_value(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return default


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_media_path(dataset: Path, row: dict[str, str]) -> Path:
    raw = first_value(row, ("media_path", "relative_path", "path", "file", "source_path"))
    if not raw:
        raise ValueError("manifest row has no media path column")
    path = Path(raw)
    return path if path.is_absolute() else dataset / path


def resolve_model_path(model: Path) -> Path:
    if model.is_absolute():
        return model
    return (Path.cwd() / model).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_detector(model_path: Path):
    """Load the existing detector and adapt only evaluator-process globals.

    The current runtime module has a module-level input size.  Reading the ONNX
    input shape here keeps 256x256 evaluation compatible without editing the
    deployment source or changing its normal runtime behavior.
    """
    import cv2
    import onnx_inference as runtime

    session = runtime.ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    input_size = next((int(value) for value in reversed(shape) if isinstance(value, int)), 640)
    original_letterbox = runtime.letterbox

    def evaluator_letterbox(frame):
        return original_letterbox(frame, new_shape=(input_size, input_size))

    runtime.IMGSZ = input_size
    runtime.letterbox = evaluator_letterbox
    detector = runtime.ONNXPoseDetector(str(model_path), imgsz=input_size)
    return detector, input_size, cv2


def event_info(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected_alarm = any(as_bool(row.get("expected_alarm")) for row in rows)
    fall_starts = [as_float(row.get("fall_start_s")) for row in rows]
    fall_starts = [value for value in fall_starts if value is not None]
    return {
        "expected_alarm": expected_alarm,
        "expected_recovery": any(as_bool(row.get("expected_recovery")) for row in rows),
        "fall_start_s": min(fall_starts) if fall_starts else None,
    }


def metric_class(expected_alarm: bool, predicted_alarm: bool) -> str:
    if expected_alarm and predicted_alarm:
        return "TP"
    if not expected_alarm and predicted_alarm:
        return "FP"
    if expected_alarm and not predicted_alarm:
        return "FN"
    return "TN"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def duration_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_s": sum(values) / len(values) if values else None,
        "median_s": percentile(values, 0.5),
        "p95_s": percentile(values, 0.95),
        "max_s": max(values) if values else None,
    }


def evaluate_video(
    row: dict[str, str],
    events: dict[str, list[dict[str, str]]],
    detector,
    fall_detector,
    cv2,
    confidence_threshold: float,
    alarm_match_window: float,
    max_frames: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_id = first_value(row, ("sample_id", "id"))
    scene = first_value(row, ("scene", "category"), "unknown")
    media_path = resolve_media_path(DATASET, row)
    metadata = event_info(events.get(sample_id, []))
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        return (
            {
                "sample_id": sample_id,
                "scene": scene,
                "media_path": str(media_path),
                "status": "INVALID",
                "error": "unable to open media",
                "expected_alarm": metadata["expected_alarm"],
                "predicted_alarm": "INVALID",
            },
            [],
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or as_float(row.get("fps")) or 30.0
    frame_index = 0
    person_frames = 0
    no_person_frames = 0
    missing_keypoint_count = 0
    missing_frame_count = 0
    alarm_time = None
    recovery_time = None
    previous_fall = False
    frame_rows = []
    error = ""

    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            result = None
            missing_names: list[int] = []
            try:
                pose_results = detector(frame)
                xy = pose_results[0].keypoints.xy
                conf = pose_results[0].keypoints.conf
                if len(xy) == 0:
                    no_person_frames += 1
                else:
                    person_frames += 1
                    person_conf = conf[0]
                    missing_names = [
                        index for index, value in enumerate(person_conf)
                        if float(value) < confidence_threshold
                    ]
                    missing_keypoint_count += len(missing_names)
                    if any(person_conf[index] < confidence_threshold for index in REQUIRED_POINTS):
                        missing_frame_count += 1
                    else:
                        result = fall_detector.detect(xy[0], timestamp)
            except Exception as exc:  # retain per-video diagnostic output
                error = f"inference failed at frame {frame_index}: {exc}"
                break

            fall = bool(result.fall) if result is not None else False
            if fall and alarm_time is None:
                alarm_time = timestamp
            if previous_fall and not fall and alarm_time is not None and recovery_time is None:
                recovery_time = timestamp
            previous_fall = fall
            frame_rows.append(
                {
                    "sample_id": sample_id,
                    "frame_index": frame_index - 1,
                    "time_s": round(timestamp, 6),
                    "state": result.state if result is not None else "NO_VALID_POSE",
                    "fall": int(fall),
                    "missing_keypoints": ";".join(map(str, missing_names)),
                    "missing_keypoint_count": len(missing_names),
                }
            )
    finally:
        capture.release()

    status = "INVALID" if error or frame_index == 0 else "OK"
    predicted_alarm = "INVALID" if status == "INVALID" else ("ALARM" if alarm_time is not None else "NO_ALARM")
    alarm_matches = (
        alarm_time is not None
        and metadata["fall_start_s"] is not None
        and metadata["fall_start_s"] <= alarm_time <= metadata["fall_start_s"] + alarm_match_window
    )
    predicted_for_metrics = (
        (alarm_matches if metadata["fall_start_s"] is not None else alarm_time is not None)
        if metadata["expected_alarm"]
        else alarm_time is not None
    )
    classification = "INVALID" if status == "INVALID" else metric_class(metadata["expected_alarm"], predicted_for_metrics)
    alarm_latency = (
        alarm_time - metadata["fall_start_s"]
        if alarm_matches and metadata["fall_start_s"] is not None
        else None
    )
    return (
        {
            "sample_id": sample_id,
            "scene": scene,
            "media_path": str(media_path),
            "status": status,
            "error": error,
            "frames": frame_index,
            "duration_s": round(frame_index / fps, 6) if frame_index else None,
            "fps": fps,
            "expected_alarm": int(metadata["expected_alarm"]),
            "expected_recovery": int(metadata["expected_recovery"]),
            "predicted_alarm": predicted_alarm,
            "classification": classification,
            "TP": int(classification == "TP"),
            "FP": int(classification == "FP"),
            "FN": int(classification == "FN"),
            "first_alarm_time_s": alarm_time,
            "first_alarm_latency_s": alarm_latency,
            "recovery_time_s": recovery_time,
            "recovery_latency_s": recovery_time - alarm_time if recovery_time is not None and alarm_time is not None else None,
            "person_frames": person_frames,
            "no_person_frames": no_person_frames,
            "missing_keypoint_count": missing_keypoint_count,
            "missing_frame_count": missing_frame_count,
            "missing_keypoint_rate": (
                missing_keypoint_count / (person_frames * 17) if person_frames else None
            ),
            "required_keypoint_invalid_rate": missing_frame_count / person_frames if person_frames else None,
        },
        frame_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["ALL"] = rows
    for row in rows:
        groups[row["scene"]].append(row)
    summary = {}
    for name, group in groups.items():
        valid = [row for row in group if row.get("classification") != "INVALID"]
        summary[name] = {
            "samples": len(group),
            "valid_samples": len(valid),
            "invalid_samples": len(group) - len(valid),
            "TP": sum(row.get("classification") == "TP" for row in valid),
            "FP": sum(row.get("classification") == "FP" for row in valid),
            "FN": sum(row.get("classification") == "FN" for row in valid),
            "TN": sum(row.get("classification") == "TN" for row in valid),
            "first_alarm_latency": duration_stats([
                row["first_alarm_latency_s"] for row in valid if row.get("first_alarm_latency_s") is not None
            ]),
            "recovery_latency": duration_stats([
                row["recovery_latency_s"] for row in valid if row.get("recovery_latency_s") is not None
            ]),
            "mean_missing_keypoint_rate": (
                sum(row["missing_keypoint_rate"] for row in valid if row.get("missing_keypoint_rate") is not None)
                / len([row for row in valid if row.get("missing_keypoint_rate") is not None])
                if any(row.get("missing_keypoint_rate") is not None for row in valid)
                else None
            ),
        }
    return summary


def main() -> int:
    global DATASET
    args = parse_args()
    DATASET = args.dataset.resolve()
    manifest_path = DATASET / "manifest.csv"
    events_path = DATASET / "labels" / "events.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    manifest = load_csv(manifest_path)
    event_rows = load_csv(events_path)
    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in event_rows:
        events[first_value(event, ("sample_id", "id"))].append(event)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or DATASET / "reports" / run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    detector, input_size, cv2 = prepare_detector(model_path)
    fall_detector_module = __import__("fall_detector")

    video_rows = []
    frame_rows = []
    for row in manifest:
        fall_detector = fall_detector_module.FallDetector()
        result, frames = evaluate_video(
            row,
            events,
            detector,
            fall_detector,
            cv2,
            args.confidence_threshold,
            args.alarm_match_window,
            args.max_frames,
        )
        video_rows.append(result)
        frame_rows.extend(frames)

    summary = summarize(video_rows)
    metadata = {
        "run_id": run_id,
        "dataset": str(DATASET),
        "manifest": str(manifest_path),
        "events": str(events_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "input_size": f"{input_size}x{input_size}",
        "platform": f"{platform.system()} / {platform.machine()}",
        "python": sys.version,
        "confidence_threshold": args.confidence_threshold,
        "alarm_match_window_s": args.alarm_match_window,
        "summary": summary,
    }
    video_fields = list(video_rows[0].keys()) if video_rows else ["sample_id"]
    write_csv(output / "video_predictions.csv", video_rows, video_fields)
    write_csv(
        output / "frame_predictions.csv",
        frame_rows,
        ["sample_id", "frame_index", "time_s", "state", "fall", "missing_keypoints", "missing_keypoint_count"],
    )
    (output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "output": str(output), "summary": summary["ALL"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
