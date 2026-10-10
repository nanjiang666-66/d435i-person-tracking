import unittest

import numpy as np

from person_vision.pose_depth import PoseDepthRecovery, observe_body, visible_regions


class PoseDepthTests(unittest.TestCase):
    def setUp(self):
        self.recovery = PoseDepthRecovery()
        self.frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        self.selected = 7
        self.now = 0.0

    def track(self, z=1.0, ident=7, cx=320, width_z=None):
        scale = 1.0 / (z if width_z is None else width_z)
        joints = np.zeros((17, 3), dtype=float)
        joints[[5, 6, 11, 12]] = [(cx+90*scale, 100, .99), (cx-90*scale, 100, .99),
                                 (cx+60*scale, 350, .99), (cx-60*scale, 350, .99)]
        return dict(id=ident, box=(cx-130*scale, 20, cx+130*scale, 440),
                    confidence=.95, keypoints=joints)

    def step(self, z=1.0, tracks=None, dt=.1, frame=None):
        self.now += dt
        tracks = [self.track(z)] if tracks is None else tracks
        self.selected, obs, reason = self.recovery.update(
            self.frame if frame is None else frame,
            np.full((480,640), round(z*1000), dtype=np.uint16),
            '16UC1', tracks, self.selected, self.now, 600.0)
        return obs, reason

    def enrol(self):
        self.assertIsNone(self.step()[0])
        self.assertIsNone(self.step()[0])
        self.assertAlmostEqual(self.step()[0].depth, 1.0)

    def test_occlusion_and_retreat_recovers_without_approaching(self):
        self.enrol()
        anchor = self.recovery.anchor
        self.assertIsNone(self.step(.4, tracks=[])[0])
        brighter = np.full_like(self.frame, (190, 180, 160))
        for _ in range(2):
            self.assertIsNone(self.step(1.7, frame=brighter)[0])
        obs, _ = self.step(1.7, frame=brighter)
        self.assertAlmostEqual(obs.depth, 1.7, places=3)
        self.assertEqual(self.recovery.anchor, anchor)

    def test_gradual_brown_occluder_cannot_poison_range(self):
        self.enrol()
        brown = np.full_like(self.frame, (35, 65, 150))
        for z in (.95, .85, .75, .6, .45, .283):
            obs, _ = self.step(z, tracks=[self.track(1.0)], frame=brown)
            self.assertIsNone(obs)
            self.assertAlmostEqual(self.recovery.last_depth, 1.0)

    def test_constant_size_near_occluder_is_not_body_motion(self):
        self.enrol()
        for _ in range(6):
            self.assertIsNone(self.step(.4, tracks=[self.track(1.0)])[0])
        self.assertAlmostEqual(self.recovery.last_depth, 1.0)

    def test_real_forward_motion_with_size_change_remains_valid(self):
        self.enrol()
        for z in (.95,.9,.85,.8,.75):
            obs, _ = self.step(z)
            self.assertIsNotNone(obs)
            self.assertAlmostEqual(obs.depth,z,places=3)

    def test_changed_id_recovers_within_three_seconds(self):
        self.enrol()
        self.step(tracks=[],dt=1.)
        for _ in range(3):
            obs, _ = self.step(1.7,tracks=[self.track(1.7,ident=9)])
        self.assertEqual(self.selected,9)
        self.assertIsNotNone(obs)

    def test_preexisting_person_cannot_become_selected(self):
        self.enrol()
        self.step(tracks=[self.track(),self.track(ident=9,cx=70)])
        for _ in range(5):
            obs, _ = self.step(tracks=[self.track(ident=9)])
            self.assertIsNone(obs)
        self.assertEqual(self.selected,7)

    def test_ambiguous_new_people_and_timeout_stay_invalid(self):
        self.enrol()
        for _ in range(4):
            obs, _ = self.step(tracks=[self.track(ident=8,cx=290),self.track(ident=9,cx=350)])
            self.assertIsNone(obs)
        self.step(tracks=[],dt=3.1)
        for _ in range(4):
            self.assertIsNone(self.step(tracks=[self.track(ident=8)])[0])

    def test_low_frame_rate_does_not_require_reconfirmation_every_frame(self):
        self.enrol()
        for _ in range(8):
            self.assertIsNotNone(self.step(dt=.25)[0])

    def test_duplicate_timestamps_do_not_confirm_depth(self):
        for _ in range(5):
            self.assertIsNone(self.step(dt=0)[0])
        self.assertIsNone(self.recovery.anchor)

    def test_uncertain_pose_and_invalid_depth_decline_measurement(self):
        t=self.track();t['keypoints'][11,2]=.2
        t['keypoints'][[5,6],2]=.2
        self.assertIsNone(self.step(tracks=[t])[0])
        t=self.track()
        for encoding, d in [('16UC1',np.zeros((480,640),np.uint16)),
                            ('32FC1',np.full((480,640),np.nan))]:
            obs,_=observe_body(self.frame,d,encoding,t['box'],t['keypoints'],.001,600.)
            self.assertIsNone(obs)

    def upper_body(self, z=1.0, confidence=.95, ident=7):
        t = self.track(z, ident=ident)
        t['keypoints'][[11,12],2] = .1
        t['confidence'] = confidence
        t['box'] = (t['box'][0], 20, t['box'][2], 260)
        return t

    def test_upper_body_initialises_without_hips(self):
        for _ in range(3):
            obs, _ = self.step(tracks=[self.upper_body()])
        self.assertIsNotNone(obs)
        self.assertEqual(obs.kind, 'shoulders')
        self.assertNotIn('hips', self.recovery.anchors)
        self.assertAlmostEqual(obs.depth, 1.0)

    def test_upper_body_retreat_after_occlusion_recovers(self):
        for _ in range(3):
            self.step(tracks=[self.upper_body()])
        anchor = self.recovery.anchors['shoulders']
        self.step(tracks=[],dt=1.)
        for _ in range(3):
            obs, _ = self.step(1.7,tracks=[self.upper_body(1.7)])
        self.assertIsNotNone(obs)
        self.assertAlmostEqual(obs.depth,1.7,places=3)
        self.assertEqual(self.recovery.anchors['shoulders'],anchor)

    def test_visible_region_switch_uses_separate_body_width_references(self):
        self.enrol()
        hip_anchor = self.recovery.anchors['hips']
        shoulder_anchor = self.recovery.anchors['shoulders']
        self.assertNotEqual(hip_anchor,shoulder_anchor)
        obs, _ = self.step(tracks=[self.upper_body()])
        self.assertIsNotNone(obs)
        self.assertEqual(obs.kind,'shoulders')
        self.assertAlmostEqual(obs.depth,1.0)
        self.assertEqual(self.recovery.anchors['hips'],hip_anchor)
        self.assertEqual(self.recovery.anchors['shoulders'],shoulder_anchor)

    def test_weak_selected_detection_requires_longer_confirmation(self):
        for _ in range(4):
            self.assertIsNone(self.step(tracks=[self.upper_body(confidence=.45)])[0])
        obs,_=self.step(tracks=[self.upper_body(confidence=.45)])
        self.assertIsNotNone(obs)
        self.step(tracks=[],dt=1.)
        for _ in range(6):
            obs,_=self.step(tracks=[self.upper_body(confidence=.45,ident=9)])
            self.assertIsNone(obs)
        self.assertEqual(self.selected,7)

    def test_upper_body_occluder_does_not_poison_reference(self):
        for _ in range(3):
            self.step(tracks=[self.upper_body()])
        brown=np.full_like(self.frame,(35,65,150))
        for z in (.95,.85,.75,.60,.4,.283):
            obs,_=self.step(z,tracks=[self.upper_body()],frame=brown)
            self.assertIsNone(obs)
            self.assertAlmostEqual(self.recovery.last_depth,1.0)

    def test_diagnostics_distinguish_missing_depth_from_body_width_rejection(self):
        track = self.upper_body()
        diagnostic = {}
        for millimetres, gate in ((0, 'raw_depth_support'), (3000, 'projected_body_width')):
            obs, reason = observe_body(
                self.frame, np.full((480,640), millimetres, dtype=np.uint16),
                '16UC1', track['box'], track['keypoints'], .001, 600.,
                diagnostic=diagnostic)
            self.assertIsNone(obs)
            self.assertEqual(reason, 'body_depth_inconsistent')
            self.assertEqual(diagnostic['gate'], gate)
            self.assertEqual(diagnostic['region'], 'shoulders')

    def test_diagnostic_reports_patch_disagreement_without_changing_reference(self):
        self.enrol()
        track = self.upper_body()
        depth = np.zeros((480,640), dtype=np.uint16)
        depth[136:171] = 1000
        diagnostic = {}
        obs, reason = observe_body(
            self.frame, depth, '16UC1', track['box'], track['keypoints'], .001, 600.,
            self.recovery.anchors, 'shoulders', diagnostic)
        self.assertIsNone(obs)
        self.assertEqual(reason, 'torso_partially_occluded')
        self.assertEqual(diagnostic['gate'], 'patch_agreement')
        self.assertLess(diagnostic['supported_patches'], 4)
        self.assertAlmostEqual(self.recovery.last_depth, 1.)

    def face_track(self, z=.7, body_z=1.):
        t = self.upper_body(body_z)
        t['keypoints'][:3] = [(320,70,.99), (320+18/z,55,.99), (320-18/z,55,.99)]
        return t

    def regional_step(self, body_z, face_z, face_visible=True, background=False):
        self.now += .1
        t = self.face_track(face_z)
        depth = np.full((480,640), round(body_z*1000), np.uint16)
        frame = np.full_like(self.frame, 200)
        face_roi = next(roi for roi, _, kind in visible_regions(
            t['box'], t['keypoints'], depth.shape) if kind == 'face')
        x1,y1,x2,y2 = face_roi
        depth[y1:y2,x1:x2] = round(face_z*1000)
        frame[y1:y2,x1:x2] = (70,110,175)
        if background:
            depth[y1:y1+(y2-y1)//3,x1:x2] = 5000
        if not face_visible:
            t['keypoints'][:3,2] = .1
        self.selected, obs, reason = self.recovery.update(
            frame, depth, '16UC1', [t], self.selected, self.now, 600.)
        return obs, reason

    def test_face_initialises_while_chest_is_occluded_without_hips(self):
        for _ in range(3):
            obs,_ = self.regional_step(.36,.7)
        self.assertEqual(obs.kind,'face')
        self.assertAlmostEqual(obs.depth,.7,places=3)
        self.assertEqual(set(self.recovery.anchors), {'face'})

    def test_face_and_chest_have_independent_depth_and_colour_references(self):
        for _ in range(3):
            obs,_ = self.regional_step(1.,.7)
        self.assertEqual(obs.kind,'shoulders')
        # Eye pixel span must be multiplied by face depth, not chest depth.
        self.assertAlmostEqual(self.recovery.anchors['face'],36.,places=3)
        self.assertGreater(np.max(np.abs(self.recovery.appearances['face'][0]
                                       - self.recovery.appearances['shoulders'][0])),.12)
        for _ in range(3):
            obs,_ = self.regional_step(.36,.7)
        self.assertEqual(obs.kind,'face')
        self.assertAlmostEqual(obs.depth,.7,places=3)
        self.assertEqual(self.recovery.anchors['shoulders'],180.)

    def test_plausible_foreground_chest_is_excluded_when_face_is_visible(self):
        # .5m passes the broad shoulder-width gate, but is in front of the
        # .7m face. Even a white object must not initialise as the torso.
        for _ in range(3):
            obs,_ = self.regional_step(.5,.7)
        self.assertEqual(obs.kind,'face')
        self.assertNotIn('shoulders',self.recovery.anchors)

    def test_face_background_pixels_do_not_replace_face_surface(self):
        for _ in range(3):
            obs,_ = self.regional_step(.36,.7,background=True)
        self.assertEqual(obs.kind,'face')
        self.assertAlmostEqual(obs.depth,.7,places=3)
        self.assertGreater(obs.v,55.)

    def test_face_unavailable_does_not_measure_occluder_or_change_reference(self):
        for _ in range(3):
            self.regional_step(.36,.7)
        anchor = self.recovery.anchors['face']
        for _ in range(5):
            self.assertIsNone(self.regional_step(.36,.7,face_visible=False)[0])
        self.assertEqual(self.recovery.anchors,{'face':anchor})
        self.assertAlmostEqual(self.recovery.last_depth,.7,places=3)
        for _ in range(3):
            obs,_ = self.regional_step(.36,1.3)
        self.assertEqual(obs.kind,'face')
        self.assertAlmostEqual(obs.depth,1.3,places=3)

    def test_only_head_in_frame_has_region_without_shoulders(self):
        t = self.face_track()
        t['keypoints'][[5,6],2] = .1
        t['box'] = (220,20,420,120)
        obs,_ = observe_body(self.frame,np.full((480,640),700,np.uint16),
                             '16UC1',t['box'],t['keypoints'],.001,600.)
        self.assertEqual(obs.kind,'face')
        t['keypoints'][0,0] = 2
        self.assertIsNone(observe_body(self.frame,np.full((480,640),700,np.uint16),
                                      '16UC1',t['box'],t['keypoints'],.001,600.)[0])

    def test_first_visible_face_after_body_loss_can_confirm_at_new_range(self):
        for _ in range(3):
            self.step(tracks=[self.upper_body()])
        self.step(tracks=[],dt=1.)
        t = self.face_track(1.7)
        t['keypoints'][[5,6],2] = .1
        for _ in range(4):
            self.assertIsNone(self.step(1.7,tracks=[t])[0])
        obs,_ = self.step(1.7,tracks=[t])
        self.assertEqual(obs.kind,'face')
        self.assertAlmostEqual(obs.depth,1.7,places=3)
        self.assertEqual(self.recovery.anchors['shoulders'],180.)

    def test_sample_ray_is_a_supported_pixel_even_with_central_depth_hole(self):
        t = self.upper_body()
        depth = np.full((480,640),1000,np.uint16)
        depth[174:198,308:332] = 0
        obs,_ = observe_body(self.frame,depth,'16UC1',t['box'],t['keypoints'],.001,600.)
        self.assertIsNotNone(obs)
        self.assertEqual(depth[int(obs.v),int(obs.u)],1000)


if __name__ == '__main__':
    unittest.main()
