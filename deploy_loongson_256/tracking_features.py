"""Small, versioned tracking messages for the independent motion process."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any

import numpy as np


@dataclass
class TrackingFeatures:
    schema_version: int
    sequence: int
    timestamp: float
    frame_width: int
    frame_height: int
    target_valid: bool
    target_count: int
    confidence: float
    bbox: list[float] | None
    center_x: float | None
    center_y: float | None
    horizontal_error: float | None
    distance_ratio: float | None
    fall_state: str
    fall: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")


def extract_tracking_features(
    pose_result: Any,
    frame_shape: tuple[int, ...],
    detection: Any = None,
    sequence: int = 0,
    timestamp: float | None = None,
) -> TrackingFeatures:
    """Convert a pose result to the small message consumed by motion control."""
    frame_height, frame_width = frame_shape[:2]
    boxes = np.asarray(pose_result.boxes, dtype=np.float32)
    keypoints = pose_result.keypoints
    xy = np.asarray(keypoints.xy, dtype=np.float32)
    conf = np.asarray(keypoints.conf, dtype=np.float32)
    target_count = len(boxes)
    required = (5, 6, 11, 12)
    valid = (
        target_count > 0
        and len(xy) > 0
        and xy.shape[1:] == (17, 2)
        and conf.shape[1:] == (17,)
        and all(conf[0, index] >= 0.5 for index in required)
    )

    state = getattr(detection, "state", "UNKNOWN") if detection is not None else "UNKNOWN"
    state = getattr(state, "name", str(state))
    fall = bool(getattr(detection, "fall", False)) if detection is not None else False
    if not valid:
        return TrackingFeatures(
            1, sequence, time.monotonic() if timestamp is None else timestamp,
            frame_width, frame_height, False, target_count, 0.0,
            None, None, None, None, None, state, fall,
        )

    box = boxes[0]
    x1, y1, x2, y2 = [float(value) for value in box]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    confidence = float(np.min(conf[0, list(required)]))
    return TrackingFeatures(
        1, sequence, time.monotonic() if timestamp is None else timestamp,
        frame_width, frame_height, True, target_count, confidence,
        [x1, y1, x2, y2], center_x, center_y,
        (center_x / frame_width) * 2.0 - 1.0,
        max(0.0, (y2 - y1) / frame_height), state, fall,
    )


class UDPFeatureSender:
    """Send the latest tracking features to a local motion process."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9001):
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        import socket

        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, features: TrackingFeatures) -> None:
        self.socket.sendto(features.to_json(), self.address)

    def close(self) -> None:
        self.socket.close()
