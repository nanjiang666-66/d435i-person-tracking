import unittest

import numpy as np

from person_vision.depth_estimation import distance_in_box


class DepthEstimationTests(unittest.TestCase):
    def test_leaning_person_keeps_torso_depth(self):
        depth = np.full((100, 100), 1300, dtype=np.uint16)
        # Foreground torso occupies only part of the new lower-middle ROI.
        depth[60:85, 25:50] = 850
        self.assertAlmostEqual(distance_in_box(depth, "16UC1", (0, 0, 100, 100), 0.001), 0.85, places=3)

    def test_few_near_outliers_do_not_override_person(self):
        depth = np.full((100, 100), 850, dtype=np.uint16)
        depth[40:44, 25:45] = 400
        self.assertAlmostEqual(distance_in_box(depth, "16UC1", (0, 0, 100, 100), 0.001), 0.85, places=3)

    def test_invalid_and_unsupported_depth(self):
        depth = np.zeros((100, 100), dtype=np.uint16)
        self.assertIsNone(distance_in_box(depth, "16UC1", (0, 0, 100, 100), 0.001))
        self.assertIsNone(distance_in_box(np.ones((100, 100), dtype=np.float32), "rgb8", (0, 0, 100, 100), 0.001))

    def test_float_depth_and_spread_foreground(self):
        depth = np.full((100, 100), 1.3, dtype=np.float32)
        depth[40:85, 25:45] = 0.87
        depth[40:85, 45:50] = 0.83
        self.assertAlmostEqual(distance_in_box(depth, "32FC1", (0, 0, 100, 100), 0.001), 0.87, places=2)

    def test_chest_occluder_does_not_replace_locked_person(self):
        depth = np.full((100, 100), 980, dtype=np.uint16)
        # Occluder is the majority, but enough torso pixels remain visible.
        depth[40:85, 25:58] = 590
        self.assertAlmostEqual(distance_in_box(depth, "16UC1", (0, 0, 100, 100), 0.001, reference_depth=0.98), 0.98, places=2)

    def test_full_occlusion_has_no_trusted_depth(self):
        depth = np.full((100, 100), 590, dtype=np.uint16)
        self.assertIsNone(distance_in_box(depth, "16UC1", (0, 0, 100, 100), 0.001, reference_depth=0.98))


if __name__ == "__main__":
    unittest.main()
