"""Local smoke tests for the server inference service."""
from __future__ import annotations

import io
import unittest

import cv2
import numpy as np

from server_inference import InferenceService, create_app


class FakeInput:
    name = "images"


class FakeOutput:
    name = "output0"


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeOutput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        self.last_inputs = inputs
        return [np.zeros((1, 56, 1), dtype=np.float32)]


def fake_preprocess(frame, imgsz):
    return np.zeros((1, 3, imgsz, imgsz), dtype=np.float32), 1.0, (0.0, 0.0)


def fake_postprocess(output, orig_shape, ratio, padding, imgsz):
    return (
        np.array([[1, 2, 30, 40]], dtype=np.float32),
        np.zeros((1, 17, 3), dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )


class ServerInferenceSmokeTests(unittest.TestCase):
    def setUp(self):
        self.service = InferenceService(
            model_path=__file__,
            session=FakeSession(),
            preprocess_fn=fake_preprocess,
            postprocess_fn=fake_postprocess,
        )

    def test_infer_bytes_returns_json_fields(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        result = self.service.infer_bytes(encoded.tobytes())
        self.assertEqual(result["boxes"], [[1.0, 2.0, 30.0, 40.0]])
        self.assertEqual(len(result["keypoints"]), 1)
        self.assertEqual(result["scores"], [0.8999999761581421])
        self.assertGreaterEqual(result["inference_time_ms"], 0.0)

    def test_fastapi_app_exposes_infer_route_when_installed(self):
        try:
            app = create_app(self.service)
        except RuntimeError as exc:
            if "FastAPI" in str(exc):
                self.skipTest(str(exc))
            raise
        routes = {(route.path, tuple(route.methods or ())) for route in app.routes}
        self.assertTrue(any(path == "/health" and "GET" in methods for path, methods in routes))
        self.assertTrue(any(path == "/infer" and "POST" in methods for path, methods in routes))


if __name__ == "__main__":
    unittest.main()
