"""RGB-D association regressions for arbitrary visible parts and occluders."""
import unittest

import numpy as np

from person_vision.segment_depth import SegmentDepthRecovery, person_surfaces


class SegmentDepthTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((160, 200, 3), (220, 220, 220), np.uint8)
        self.depth = np.full((160, 200), 1000, np.uint16)
        self.mask = np.zeros((160, 200), bool)
        self.mask[10:150, 60:145] = True
        self.recovery = SegmentDepthRecovery()
        self.now = 0.
        self.ident = 7

    def step(self, ident=7, confidence=.95, missing=False, dt=.1):
        self.now += dt
        tracks = [] if missing else [dict(id=ident, box=(0, 0, 200, 160),
                                         confidence=confidence, mask=self.mask)]
        self.ident, observation, reason = self.recovery.update(
            self.frame, self.depth, '16UC1', tracks, self.ident, self.now)
        return observation, reason

    def warm(self):
        observation = None
        for _ in range(4):
            observation, _ = self.step()
        self.assertIsNotNone(observation)

    def test_head_or_legs_alone_without_any_landmarks_can_initialize(self):
        for top, bottom in ((10, 45), (110, 155)):
            with self.subTest(top=top):
                self.recovery.reset()
                self.mask[:] = False
                self.mask[top:bottom, 75:130] = True
                self.warm()
                self.assertAlmostEqual(self.recovery.last_depth, 1.)

    def test_narrow_side_view_does_not_need_frontal_body_width(self):
        self.mask[:] = False
        self.mask[5:155, 110:130] = True
        self.warm()
        observation, _ = self.step()
        self.assertAlmostEqual(observation.depth, 1.)

    def test_unknown_trousers_can_replace_upper_body_at_same_range(self):
        self.warm()
        self.mask[:] = False
        self.mask[110:155, 80:130] = True
        self.frame[:] = (90, 25, 20)
        for _ in range(4):
            observation, _ = self.step()
            self.assertIsNone(observation)
        observation, _ = self.step()
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(observation.depth, 1.)

    def test_gradually_approaching_occluder_does_not_poison_range(self):
        self.warm()
        bank = [sig.copy() for sig in self.recovery.bank]
        # The model still labels the obstructed ROI as person: the depth
        # selector must not follow the object one small step at a time.
        self.frame[:] = (25, 60, 180)
        for z in (.96, .91, .85, .78, .7, .6, .5, .4, .283):
            self.depth[:] = round(z * 1000)
            observation, _ = self.step()
            self.assertIsNone(observation)
        self.assertAlmostEqual(self.recovery.last_depth, 1.)
        self.assertEqual(len(bank), len(self.recovery.bank))
        for first, second in zip(bank, self.recovery.bank):
            np.testing.assert_array_equal(first, second)

    def test_obstacle_in_person_mask_keeps_remaining_body_surface(self):
        self.warm()
        self.depth[45:120, 70:140] = 350
        self.frame[45:120, 70:140] = (30, 70, 180)
        observation, _ = self.step()
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(observation.depth, 1.)
        self.assertEqual(self.depth[int(observation.v), int(observation.u)], 1000)

    def test_full_occlusion_has_no_measured_position(self):
        self.warm()
        self.mask[:] = False
        self.depth[:] = 300
        for _ in range(10):
            observation, _ = self.step()
            self.assertIsNone(observation)
        self.assertAlmostEqual(self.recovery.last_depth, 1.)

    def test_similar_colour_foreground_does_not_outvote_remaining_person(self):
        self.warm()
        self.depth[15:145, 62:135] = 350
        # Segmentation mistakenly includes a near object with the SAME colour.
        # Remaining person pixels at the tracked range should win over size.
        observation, _ = self.step()
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(observation.depth, 1.)

    def test_two_similar_new_people_remain_ambiguous(self):
        self.warm()
        self.step(missing=True, dt=.5)
        tracks = []
        for ident, left in [(8, 35), (9, 105)]:
            mask = np.zeros_like(self.mask)
            mask[10:150, left:left+55] = True
            tracks.append(dict(id=ident, box=(left,10,left+55,150), confidence=.95, mask=mask))
        for _ in range(6):
            self.now += .1
            ident, observation, reason = self.recovery.update(
                self.frame, self.depth, '16UC1', tracks, 7, self.now)
            self.assertIsNone(observation)
            self.assertEqual(ident, 7)
            self.assertEqual(reason, 'ambiguous_identity')

    def test_retreat_after_occlusion_recovers_without_returning_to_old_range(self):
        self.warm()
        self.step(missing=True, dt=.7)
        self.depth[:] = 1700
        for _ in range(4):
            observation, _ = self.step()
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(observation.depth, 1.7)

    def test_normal_person_motion_in_both_directions_remains_measurable(self):
        self.warm()
        for z in (1.05, 1.1, 1.15, 1.1, 1., .95, .9):
            self.depth[:] = round(z * 1000)
            observation, _ = self.step()
            self.assertIsNotNone(observation)
            self.assertAlmostEqual(observation.depth, z, places=3)

    def test_changed_id_confirmation_and_timeout_are_separate(self):
        self.warm()
        self.step(missing=True, dt=1.)
        for _ in range(4):
            observation, _ = self.step(ident=8)
            self.assertIsNone(observation)
        observation, _ = self.step(ident=8)
        self.assertIsNotNone(observation)
        self.assertEqual(self.ident, 8)
        self.step(missing=True, dt=3.1)
        observation, reason = self.step(ident=9)
        self.assertIsNone(observation)
        self.assertEqual(reason, 'target_missing')

    def test_depth_holes_and_foreground_excluded_pixels_do_not_set_sample_ray(self):
        self.depth[60:100, 85:120] = 0
        surfaces, _, _ = person_surfaces(self.frame, self.depth, '16UC1',
                                         (0,0,200,160), self.mask)
        self.assertTrue(surfaces)
        surface = surfaces[0]
        self.assertEqual(self.depth[int(surface.v), int(surface.u)], 1000)
        self.depth[10:150, 60:95] = 350
        excluded = np.zeros_like(self.mask)
        excluded[:, :95] = True
        surfaces, _, _ = person_surfaces(self.frame, self.depth, '16UC1',
                                         (0,0,200,160), self.mask, excluded=excluded)
        self.assertTrue(all(abs(s.depth-1.) < .01 for s in surfaces))

    def test_other_person_with_nested_bbox_and_disjoint_mask_is_not_reacquired(self):
        self.warm()
        other_mask = np.zeros_like(self.mask)
        other_mask[10:150, 5:45] = True
        target = dict(id=7, box=(0,0,200,160), confidence=.95, mask=self.mask)
        other = dict(id=9, box=(0,0,50,160), confidence=.95, mask=other_mask)
        self.now += .1
        self.recovery.update(self.frame, self.depth, '16UC1', [target,other], 7, self.now)
        self.assertIn(9, self.recovery.other_ids)
        for _ in range(8):
            self.now += .1
            ident, observation, _ = self.recovery.update(
                self.frame, self.depth, '16UC1', [other], 7, self.now)
            self.assertIsNone(observation)
            self.assertEqual(ident, 7)

    def test_white_fragment_in_dark_new_mask_does_not_change_identity(self):
        self.warm()
        self.frame[:] = 30
        self.frame[20:30,80:100] = 220
        for _ in range(8):
            observation, reason = self.step(ident=9)
            self.assertIsNone(observation)
            self.assertEqual(reason, 'identity_appearance_mismatch')
        self.assertEqual(self.ident, 7)

    def test_mask_misalignment_is_rejected_instead_of_sampling_shifted_depth(self):
        surfaces, _, reason = person_surfaces(self.frame, self.depth, '16UC1',
                                              (0,0,200,160), self.mask[::2,::2])
        self.assertFalse(surfaces)
        self.assertEqual(reason, 'mask_alignment_error')

    def test_float_depth_and_repeated_stamps(self):
        surfaces, _, _ = person_surfaces(self.frame, self.depth.astype(np.float32)/1000,
                                         '32FC1', (0,0,200,160), self.mask)
        self.assertAlmostEqual(surfaces[0].depth, 1.)
        self.warm()
        observation, reason = self.step(dt=0.)
        self.assertIsNone(observation)
        self.assertEqual(reason, 'stale_frame')


if __name__ == '__main__':
    unittest.main()
