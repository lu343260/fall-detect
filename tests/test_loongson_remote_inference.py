import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import cv2
import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy_loongson_256"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from onnx_inference import RemoteInferenceError, RemotePoseDetector


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def valid_payload():
    keypoints = np.zeros((1, 17, 3), dtype=float).tolist()
    keypoints[0][5] = [10, 20, 0.9]
    keypoints[0][6] = [20, 20, 0.9]
    keypoints[0][11] = [10, 40, 0.9]
    keypoints[0][12] = [20, 40, 0.9]
    return {
        "boxes": [[1, 2, 30, 40]],
        "keypoints": keypoints,
        "scores": [0.95],
        "inference_time_ms": 12.5,
    }


class RemotePoseDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = RemotePoseDetector("http://example.test/infer", timeout=2)
        self.frame = np.zeros((32, 48, 3), dtype=np.uint8)

    def test_fake_http_response_is_converted_to_pose_result(self):
        response = FakeResponse(json.dumps(valid_payload()).encode("utf-8"))
        with patch("onnx_inference.urllib_request.urlopen", return_value=response) as urlopen:
            result = self.detector(self.frame)[0]

        request = urlopen.call_args.args[0]
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 2)
        self.assertEqual(request.full_url, "http://example.test/infer")
        self.assertIn(b' name="file"; filename="frame.jpg"', request.data)
        self.assertIn(b"Content-Type: image/jpeg", request.data)
        self.assertEqual(result.boxes.shape, (1, 4))
        self.assertEqual(result.keypoints.xy.shape, (1, 17, 2))
        self.assertEqual(result.keypoints.conf.shape, (1, 17))
        self.assertEqual(result.inference_time_ms, 12.5)
        self.assertTrue(cv2.imdecode(np.frombuffer(request.data.split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0], dtype=np.uint8), cv2.IMREAD_COLOR) is not None)

    def test_invalid_json_is_reported(self):
        with patch("onnx_inference.urllib_request.urlopen", return_value=FakeResponse(b"not-json")):
            with self.assertRaisesRegex(RemoteInferenceError, "invalid JSON"):
                self.detector(self.frame)

    def test_connection_failure_is_reported(self):
        with patch(
            "onnx_inference.urllib_request.urlopen",
            side_effect=URLError("offline"),
        ):
            with self.assertRaisesRegex(RemoteInferenceError, "request failed"):
                self.detector(self.frame)

    def test_http_server_error_is_reported(self):
        response = FakeResponse(b'{"detail": "not ready"}', status=503)
        with patch("onnx_inference.urllib_request.urlopen", return_value=response):
            with self.assertRaisesRegex(RemoteInferenceError, "HTTP 503"):
                self.detector(self.frame)

    def test_inconsistent_server_arrays_are_rejected(self):
        payload = valid_payload()
        payload["scores"] = []
        response = FakeResponse(json.dumps(payload).encode("utf-8"))
        with patch("onnx_inference.urllib_request.urlopen", return_value=response):
            with self.assertRaisesRegex(RemoteInferenceError, "inconsistent"):
                self.detector(self.frame)

    def test_coordinate_only_keypoints_with_wrong_scores_are_rejected(self):
        payload = valid_payload()
        payload["keypoints"] = np.zeros((1, 17, 2), dtype=float).tolist()
        payload["scores"] = []
        response = FakeResponse(json.dumps(payload).encode("utf-8"))
        with patch("onnx_inference.urllib_request.urlopen", return_value=response):
            with self.assertRaisesRegex(RemoteInferenceError, "coordinate-only"):
                self.detector(self.frame)


if __name__ == "__main__":
    unittest.main()
