import unittest
from pathlib import Path
from types import SimpleNamespace

import evaluate_scenario_set as evaluator


class FakeCapture:
    def __init__(self, frames, fps=5.0):
        self.frames = list(frames)
        self.fps = fps

    def isOpened(self):
        return True

    def get(self, property_id):
        return self.fps

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


class FakeCV2:
    CAP_PROP_FPS = 5
    ROTATE_180 = "ROTATE_180"

    def __init__(self, frames):
        self.capture = FakeCapture(frames)
        self.rotated = []

    def VideoCapture(self, path):
        return self.capture

    def rotate(self, frame, rotation):
        self.rotated.append((frame, rotation))
        return (frame, rotation)


class FakeDetector:
    def __init__(self, valid_results):
        self.valid_results = iter(valid_results)
        self.calls = []

    def __call__(self, frame):
        self.calls.append(frame)
        return [next(self.valid_results)]


class FakeFallDetector:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def detect(self, keypoints, timestamp):
        self.calls.append((keypoints, timestamp))
        return next(self.results)


def pose_result(valid=True):
    if not valid:
        return SimpleNamespace(
            keypoints=SimpleNamespace(xy=[], conf=[])
        )
    xy = [[[float(index), float(index)] for index in range(17)]]
    conf = [[1.0 for _ in range(17)]]
    return SimpleNamespace(keypoints=SimpleNamespace(xy=xy, conf=conf))


class EvaluatorPhase1Tests(unittest.TestCase):
    def setUp(self):
        evaluator.DATASET = Path(".")

    def test_rotation_and_deployment_state_update_frequency(self):
        fake_cv2 = FakeCV2(range(6))
        detector = FakeDetector([pose_result(True) for _ in range(6)])
        fall_detector = FakeFallDetector([
            SimpleNamespace(fall=True, state=2),
            SimpleNamespace(fall=True, state=2),
        ])
        row = {"sample_id": "sample-1", "scene": "fall_front", "media_path": "video.mp4"}

        result, frame_rows = evaluator.evaluate_video(
            row,
            {"sample-1": [{"expected_alarm": "true", "fall_start_s": "0"}]},
            detector,
            fall_detector,
            fake_cv2,
            0.5,
            5.0,
            None,
            frame_skip=1,
        )

        self.assertEqual(len(fake_cv2.rotated), 6)
        self.assertTrue(all(rotation == fake_cv2.ROTATE_180 for _, rotation in fake_cv2.rotated))
        self.assertEqual(len(detector.calls), 6)
        self.assertEqual(len(fall_detector.calls), 2)
        self.assertEqual([row["state_update"] for row in frame_rows], [0, 0, 1, 0, 0, 1])
        self.assertEqual(result["classification"], "TP")

    def test_frame_skip_and_invalid_frame_preserve_last_state(self):
        fake_cv2 = FakeCV2(range(6))
        detector = FakeDetector([pose_result(True), pose_result(True), pose_result(True)])
        fall_detector = FakeFallDetector([
            SimpleNamespace(fall=True, state=2),
        ])
        row = {"sample_id": "sample-2", "scene": "fall_side", "media_path": "video.mp4"}

        result, frame_rows = evaluator.evaluate_video(
            row,
            {"sample-2": [{"expected_alarm": "true", "fall_start_s": "0"}]},
            detector,
            fall_detector,
            fake_cv2,
            0.5,
            5.0,
            None,
            frame_skip=2,
        )

        # Deployment conditions require both frame 6 % FRAME_SKIP == 0 and
        # frame 6 % 3 == 0, so only the sixth frame updates the state machine.
        self.assertEqual(len(detector.calls), 3)
        self.assertEqual(len(fall_detector.calls), 1)
        self.assertEqual([row["inference_run"] for row in frame_rows], [0, 1, 0, 1, 0, 1])
        self.assertEqual(result["recovery_time_s"], None)
        self.assertEqual(result["predicted_alarm"], "ALARM")

    def test_invalid_inference_does_not_trigger_recovery(self):
        fake_cv2 = FakeCV2(range(6))
        detector = FakeDetector([pose_result(True) for _ in range(5)] + [pose_result(False)])
        fall_detector = FakeFallDetector([
            SimpleNamespace(fall=True, state=2),
        ])
        row = {"sample_id": "sample-3", "scene": "occlusion", "media_path": "video.mp4"}

        result, frame_rows = evaluator.evaluate_video(
            row,
            {"sample-3": [{"expected_alarm": "true", "fall_start_s": "0"}]},
            detector,
            fall_detector,
            fake_cv2,
            0.5,
            5.0,
            None,
            frame_skip=1,
        )

        # The invalid sixth frame must preserve the last valid fall state.
        self.assertEqual(len(fall_detector.calls), 1)
        self.assertEqual(result["recovery_time_s"], None)
        self.assertEqual(frame_rows[-1]["fall"], 1)
        self.assertEqual(frame_rows[-1]["state_update"], 0)


if __name__ == "__main__":
    unittest.main()
