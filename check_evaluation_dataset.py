"""Validate the structure and metadata of datasets/fall_eval_v1.

This checker uses only the Python standard library. It does not run inference,
modify videos, or change any model, detector, or deployment code.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


MANIFEST_FIELDS = ["sample_id", "video_path", "scene_type", "fps", "duration", "person_id"]
EVENT_FIELDS = ["sample_id", "start_time", "end_time", "expected_alarm"]
REQUIRED_SCENES = {
    "normal_standing", "sitting", "bending", "front_fall",
    "side_fall", "slow_fall", "recovery", "occlusion",
}
BOOLEAN_VALUES = {"true", "false"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def parse_nonnegative(value: str, field: str, row_number: int, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: {field} must be numeric, got {value!r}")
        return None
    if number < 0:
        errors.append(f"row {row_number}: {field} must be >= 0")
    return number


def validate(dataset: Path) -> list[str]:
    errors: list[str] = []
    for directory in (dataset / "videos", dataset / "labels", dataset / "reports"):
        if not directory.is_dir():
            errors.append(f"missing directory: {directory}")

    manifest_path = dataset / "manifest.csv"
    events_path = dataset / "labels" / "events.csv"
    manifest_header, manifest_rows = read_csv(manifest_path)
    event_header, event_rows = read_csv(events_path)
    if manifest_header != MANIFEST_FIELDS:
        errors.append(f"manifest.csv fields must be {MANIFEST_FIELDS}, got {manifest_header}")
    if event_header != EVENT_FIELDS:
        errors.append(f"events.csv fields must be {EVENT_FIELDS}, got {event_header}")
    if not manifest_rows:
        errors.append("manifest.csv contains no samples")
    if not event_rows:
        errors.append("events.csv contains no labels")

    sample_ids: set[str] = set()
    scene_types: set[str] = set()
    for row_number, row in enumerate(manifest_rows, start=2):
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            errors.append(f"manifest row {row_number}: sample_id is empty")
        elif sample_id in sample_ids:
            errors.append(f"manifest row {row_number}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)

        video_value = row.get("video_path", "").strip()
        if not video_value:
            errors.append(f"manifest row {row_number}: video_path is empty")
        else:
            video_path = Path(video_value)
            resolved = (dataset / video_path).resolve() if not video_path.is_absolute() else video_path.resolve()
            if dataset.resolve() not in resolved.parents:
                errors.append(f"manifest row {row_number}: video_path escapes dataset: {video_value!r}")
            elif not resolved.is_file():
                errors.append(f"manifest row {row_number}: video file not found: {resolved}")

        scene_type = row.get("scene_type", "").strip()
        if scene_type:
            scene_types.add(scene_type)
        else:
            errors.append(f"manifest row {row_number}: scene_type is empty")
        fps = parse_nonnegative(row.get("fps", ""), "fps", row_number, errors)
        if fps == 0:
            errors.append(f"manifest row {row_number}: fps must be > 0")
        parse_nonnegative(row.get("duration", ""), "duration", row_number, errors)
        if not row.get("person_id", "").strip():
            errors.append(f"manifest row {row_number}: person_id is empty")

    missing_scenes = REQUIRED_SCENES - scene_types
    if missing_scenes:
        errors.append(f"required scene_type values missing: {sorted(missing_scenes)}")

    event_sample_ids: set[str] = set()
    for row_number, row in enumerate(event_rows, start=2):
        sample_id = row.get("sample_id", "").strip()
        event_sample_ids.add(sample_id)
        if not sample_id:
            errors.append(f"events row {row_number}: sample_id is empty")
        elif sample_id not in sample_ids:
            errors.append(f"events row {row_number}: sample_id {sample_id!r} not found in manifest")
        start = parse_nonnegative(row.get("start_time", ""), "start_time", row_number, errors)
        end = parse_nonnegative(row.get("end_time", ""), "end_time", row_number, errors)
        if start is not None and end is not None and end < start:
            errors.append(f"events row {row_number}: end_time must be >= start_time")
        if row.get("expected_alarm", "").strip().lower() not in BOOLEAN_VALUES:
            errors.append(f"events row {row_number}: expected_alarm must be true or false")

    for sample_id in sorted(sample_ids - event_sample_ids):
        errors.append(f"missing event label for manifest sample_id {sample_id!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check fall_eval_v1 dataset metadata and files.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/fall_eval_v1"))
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    errors = validate(dataset)
    if errors:
        print(f"DATASET_INVALID: {dataset}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"DATASET_OK: {dataset}")
    print(f"samples: {len(read_csv(dataset / 'manifest.csv')[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
