import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy_loongson_256"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from tracking_features import extract_tracking_features


class TrackingFeaturesTests(unittest.TestCase):
    def test_valid_pose_is_reduced_to_follow_features(self):
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        conf = np.ones((1, 17), dtype=np.float32)
        pose = SimpleNamespace(
            boxes=np.array([[100, 20, 300, 420]], dtype=np.float32),
            keypoints=SimpleNamespace(xy=xy, conf=conf),
        )
        features = extract_tracking_features(pose, (480, 640, 3), sequence=7, timestamp=1.5)
        self.assertTrue(features.target_valid)
        self.assertEqual(features.sequence, 7)
        self.assertEqual(features.center_x, 200.0)
        self.assertAlmostEqual(features.horizontal_error, -0.375)
        self.assertAlmostEqual(features.distance_ratio, 400 / 480)
        self.assertEqual(features.to_dict()["schema_version"], 1)

    def test_missing_required_keypoint_marks_target_invalid(self):
        pose = SimpleNamespace(
            boxes=np.array([[100, 20, 300, 420]], dtype=np.float32),
            keypoints=SimpleNamespace(
                xy=np.zeros((1, 17, 2), dtype=np.float32),
                conf=np.zeros((1, 17), dtype=np.float32),
            ),
        )
        features = extract_tracking_features(pose, (480, 640, 3))
        self.assertFalse(features.target_valid)
        self.assertIsNone(features.bbox)


if __name__ == "__main__":
    unittest.main()
