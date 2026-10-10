"""Exercise the pose branch of the actual ROS callback with fake adapters."""

from types import SimpleNamespace
import unittest

import numpy as np

import test_tracker_callback as callback_fixture
from test_tracker_callback import Tensor
from person_vision.pose_depth import PoseDepthRecovery


class PoseCallbackTests(unittest.TestCase):
    def setUp(self):
        callback_fixture.TrackerCallbackTests.setUpClass()
        self.case = callback_fixture.TrackerCallbackTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.node = self.case.node
        self.node.pose_depth_enabled = True
        self.node.pose_recovery = PoseDepthRecovery()
        self.node.depth_debug = True
        self.node.intrinsics = (600., 600., 320., 240.)
        for _ in range(3):
            self.frame()
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)

    def frame(self, z=1., ident=7, width_z=None, colour=(100,100,100),
              hips_visible=True, confidence=.95):
        self.case.now += .1
        self.node.point_pub.reset_mock()
        scale = 1 / (z if width_z is None else width_z)
        joints = np.zeros((17,3))
        joints[[5,6,11,12]] = [(320+90*scale,100,.99),(320-90*scale,100,.99),
                                (320+60*scale,350,.99),(320-60*scale,350,.99)]
        if not hips_visible:
            joints[[11,12],2]=.05
        boxes = None if ident is None else SimpleNamespace(
            xyxy=Tensor([[320-130*scale,20,320+130*scale,440]]),
            id=Tensor([ident]),conf=Tensor([confidence]))
        result = SimpleNamespace(boxes=boxes,keypoints=SimpleNamespace(data=Tensor([joints])))
        self.node.model.track.return_value = [result]
        seconds=self.case.now
        header=SimpleNamespace(stamp=SimpleNamespace(sec=int(seconds),nanosec=int(seconds%1*1e9)))
        self.case.module.PersonTracker.on_frame(self.node,
            SimpleNamespace(pixels=np.full((480,640,3),colour,np.uint8),header=header),
            SimpleNamespace(pixels=np.full((480,640),round(z*1000),np.uint16),encoding='16UC1'))

    def test_xyz_uses_depth_sample_ray_and_remains_finite(self):
        self.node.show_window = True
        self.frame()
        point=self.node.point_pub.publish.call_args.args[0].point
        self.assertTrue(np.isfinite([point.x,point.y,point.z]).all())
        self.assertAlmostEqual(point.z,1.)
        self.assertAlmostEqual(point.x,0.,delta=.005)
        # Torso sampling centre is above the whole detection-box centre.
        self.assertLess(point.y,-.03)
        self.case.module.cv2.circle.assert_called_once()

    def test_occluder_does_not_publish_or_modify_reference(self):
        for z in (.9,.8,.6,.4,.283):
            self.frame(z,width_z=1.,colour=(35,65,150))
            self.case.assert_invalid_without_position()
        self.assertAlmostEqual(self.node.pose_recovery.last_depth,1.)

    def test_same_id_farther_depth_recovers_and_publishes(self):
        self.frame(.4,ident=None)
        self.case.assert_invalid_without_position()
        for _ in range(3):
            self.frame(1.7,colour=(190,180,160))
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertAlmostEqual(self.node.point_pub.publish.call_args.args[0].point.z,1.7,places=3)

    def test_changed_id_updates_ros_parameter_then_keeps_reference(self):
        self.case.now += .8
        self.frame(ident=None)
        for _ in range(3):
            self.frame(1.7,ident=8)
        self.assertEqual(self.case.target_id,8)
        self.assertEqual(self.node.last_selected_id,8)
        anchor=self.node.pose_recovery.anchor
        self.frame(1.7,ident=8)
        self.assertEqual(self.node.pose_recovery.anchor,anchor)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)

    def test_disabled_reacquire_does_not_change_id(self):
        self.node.auto_reacquire=False
        for _ in range(6):
            self.frame(1.7,ident=8)
        self.assertEqual(self.case.target_id,7)
        self.case.assert_invalid_without_position()

    def test_seated_upper_body_publishes_without_hip_visibility(self):
        self.node.pose_recovery.reset()
        for _ in range(5):
            self.frame(hips_visible=False,confidence=.45)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertAlmostEqual(self.node.point_pub.publish.call_args.args[0].point.z,1.0)
        self.assertEqual(self.node.pose_recovery.last_kind,'shoulders')

    def test_head_only_publishes_face_xyz_instead_of_foreground_depth(self):
        self.node.pose_recovery.reset()
        joints = np.zeros((17,3))
        joints[:3] = [(320,100,.99),(342,80,.99),(298,80,.99)]
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=Tensor([[260,30,380,150]]),
                                  id=Tensor([7]),conf=Tensor([.95])),
            keypoints=SimpleNamespace(data=Tensor([joints])))
        self.node.model.track.return_value = [result]
        depth = np.full((480,640),350,np.uint16)
        depth[93:125,302:338] = 600
        for _ in range(3):
            self.case.now += .1
            seconds = self.case.now
            header = SimpleNamespace(stamp=SimpleNamespace(
                sec=int(seconds),nanosec=int(seconds%1*1e9)))
            self.case.module.PersonTracker.on_frame(self.node,
                SimpleNamespace(pixels=np.full((480,640,3),150,np.uint8),header=header),
                SimpleNamespace(pixels=depth,encoding='16UC1'))
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        point = self.node.point_pub.publish.call_args.args[0].point
        self.assertAlmostEqual(point.z,.6,places=3)
        self.assertAlmostEqual(point.x,0.,delta=.003)
        self.assertLess(point.y,-.12)
        self.assertEqual(self.node.pose_recovery.last_kind,'face')


if __name__ == '__main__':
    unittest.main()
