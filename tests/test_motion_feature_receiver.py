import sys
import unittest
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deploy_loongson_256"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from motion_feature_receiver import safety_stop_reason


class MotionFeatureReceiverSafetyTests(unittest.TestCase):
    def test_valid_current_target_can_continue(self):
        message = {"timestamp": 10.0, "target_valid": True, "fall": False}
        self.assertIsNone(safety_stop_reason(message, now=10.2, stale_after=0.5))

    def test_fall_requires_stop(self):
        message = {"timestamp": 10.0, "target_valid": True, "fall": True}
        self.assertEqual(
            safety_stop_reason(message, now=10.2, stale_after=0.5), "fall detected"
        )

    def test_missing_or_stale_target_requires_stop(self):
        stale = {"timestamp": 10.0, "target_valid": True, "fall": False}
        missing = {"timestamp": 10.0, "target_valid": False, "fall": False}
        self.assertEqual(
            safety_stop_reason(stale, now=10.6, stale_after=0.5),
            "target missing or stale",
        )
        self.assertEqual(
            safety_stop_reason(missing, now=10.2, stale_after=0.5),
            "target missing or stale",
        )


if __name__ == "__main__":
    unittest.main()
