import unittest

import numpy as np

from person_vision.target_recovery import TargetRecovery


class TargetRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 200, 3), dtype=np.uint8)
        self.frame[10:110, 10:95] = (30, 80, 180)
        self.frame[10:110, 100:190] = (200, 80, 30)
        self.old_box = (20, 20, 70, 100)

    def make_recovery(self, others=()):
        recovery = TargetRecovery(max_gap=3.0)
        recovery.record_selected(self.frame, self.old_box, 1.0, others, 0.0)
        return recovery

    def test_short_id_change_reacquires_after_stable_frames(self):
        recovery = self.make_recovery()
        candidate = [(17, (25, 20, 75, 100), 1.05)]
        self.assertIsNone(recovery.find_match(self.frame, candidate, 0.3))
        self.assertIsNone(recovery.find_match(self.frame, candidate, 0.35))
        self.assertEqual(recovery.find_match(self.frame, candidate, 0.4)[0], 17)

    def test_person_already_present_is_not_selected(self):
        recovery = self.make_recovery(others=(9,))
        candidate = [(9, (25, 20, 75, 100), 1.0)]
        for now in (0.3, 0.35, 0.4):
            self.assertIsNone(recovery.find_match(self.frame, candidate, now))

    def test_similar_candidates_remain_ambiguous(self):
        recovery = self.make_recovery()
        candidates = [
            (17, (25, 20, 75, 100), 1.0),
            (18, (30, 20, 80, 100), 1.0),
        ]
        for now in (0.3, 0.35, 0.4):
            self.assertIsNone(recovery.find_match(self.frame, candidates, now))

    def test_timeout_and_depth_disagreement_prevent_reacquisition(self):
        recovery = self.make_recovery()
        candidate = [(17, (25, 20, 75, 100), 2.0)]
        for now in (0.3, 0.35, 0.4):
            self.assertIsNone(recovery.find_match(self.frame, candidate, now))
        candidate = [(17, (25, 20, 75, 100), 1.0)]
        self.assertIsNone(recovery.find_match(self.frame, candidate, 3.1))

    def test_sudden_depth_spike_is_rejected(self):
        recovery = self.make_recovery()
        self.assertFalse(recovery.depth_is_plausible(3.0, 0.1))
        self.assertFalse(recovery.depth_is_plausible(3.0, 1.5))
        self.assertTrue(recovery.depth_is_plausible(1.1, 0.1))


if __name__ == "__main__":
    unittest.main()
