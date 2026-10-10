"""Real node callback with mask inference/ROS adapters substituted."""
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np

import test_tracker_callback as fixture
from test_tracker_callback import Tensor
from person_vision.segment_depth import SegmentDepthRecovery


class SegmentCallbackTests(unittest.TestCase):
    def setUp(self):
        fixture.TrackerCallbackTests.setUpClass()
        self.case = fixture.TrackerCallbackTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.node = self.case.node
        self.node.segment_depth_enabled = True
        self.node.segment_recovery = SegmentDepthRecovery()
        self.node.detected_pub = Mock()
        self.node.depth_status_pub = Mock()
        self.node.depth_debug = True
        self.frame = np.full((100,100,3), 220, np.uint8)
        self.depth = np.full((100,100), 1000, np.uint16)
        self.mask = np.zeros((100,100), bool)
        self.mask[10:90, 20:80] = True
        for _ in range(3):
            self.process()
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)

    def process(self, ident=7, object_mask=None, wrong_shape=False):
        self.case.now += .1
        self.node.point_pub.reset_mock()
        person = self.mask[::2,::2] if wrong_shape else self.mask
        # An object comes FIRST in the model result. This tests mask-index
        # association after the node filters non-person detections.
        masks = [person] if object_mask is None else [object_mask, person]
        boxes = [[0,0,100,100]] * len(masks)
        ids = [ident] if object_mask is None else [99, ident]
        classes = [0] if object_mask is None else [39, 0]
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(boxes), id=Tensor(ids),
                                 conf=Tensor([.95]*len(ids)), cls=Tensor(classes)),
                                 masks=SimpleNamespace(data=Tensor(masks)))
        self.node.model.track.return_value = [result]
        now = self.case.now
        header = SimpleNamespace(stamp=SimpleNamespace(sec=int(now), nanosec=round(now%1*1e9)))
        self.case.module.PersonTracker.on_frame(self.node,
            SimpleNamespace(pixels=self.frame.copy(), header=header),
            SimpleNamespace(pixels=self.depth, encoding='16UC1'))

    def test_mask_depth_xyz_and_person_index_are_preserved(self):
        obj = np.zeros_like(self.mask)
        obj[:, :45] = True
        self.depth[:, :45] = 350
        self.process(object_mask=obj)
        point = self.node.point_pub.publish.call_args.args[0].point
        self.assertAlmostEqual(point.z, 1.)
        self.assertGreater(point.x, -.05)
        self.assertEqual(self.node.id_pub.publish.call_args.args[0].data, 7)
        self.assertEqual(self.node.model.track.call_args.kwargs['retina_masks'], True)
        self.assertIsNone(self.node.model.track.call_args.kwargs['classes'])

    def test_depth_unavailable_does_not_erase_selected_person_detection(self):
        self.depth[:] = 0
        self.process()
        self.case.assert_invalid_without_position()
        self.assertTrue(self.node.detected_pub.publish.call_args.args[0].data)
        self.assertEqual(self.node.depth_status_pub.publish.call_args.args[0].data, 'insufficient_mask_depth')

    def test_misaligned_mask_is_invalid_and_diagnostic_identifies_it(self):
        self.process(wrong_shape=True)
        self.case.assert_invalid_without_position()
        self.assertEqual(self.node.depth_status_pub.publish.call_args.args[0].data, 'mask_alignment_error')

    def test_nonfinite_intrinsics_does_not_publish_coordinates(self):
        self.node.intrinsics = (float('inf'),100,50,50)
        self.process()
        self.case.assert_invalid_without_position()
        self.assertEqual(self.node.depth_status_pub.publish.call_args.args[0].data, 'no_intrinsics')

    def test_inference_failure_clears_experimental_status_and_coordinates(self):
        self.node.model.track.side_effect = RuntimeError('test inference failure')
        self.process()
        self.case.assert_invalid_without_position()
        self.assertFalse(self.node.detected_pub.publish.call_args.args[0].data)
        self.assertEqual(self.node.depth_status_pub.publish.call_args.args[0].data, 'inference_error')

    def test_reacquired_parameter_does_not_reset_reference_on_next_callback(self):
        for _ in range(6):
            self.process(ident=8)
        self.assertEqual(self.case.target_id, 8)
        self.assertEqual(self.node.last_selected_id, 8)
        bank = len(self.node.segment_recovery.bank)
        self.process(ident=8)
        self.assertEqual(len(self.node.segment_recovery.bank), bank)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertEqual(self.node.id_pub.publish.call_args.args[0].data, 8)

    def test_camera_timeout_invalidates_depth_and_detection_status(self):
        self.case.now += 2.
        self.case.module.PersonTracker.check_camera_timeout(self.node)
        self.assertFalse(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertFalse(self.node.detected_pub.publish.call_args.args[0].data)
        self.assertEqual(self.node.depth_status_pub.publish.call_args.args[0].data, 'camera_timeout')


if __name__ == '__main__':
    unittest.main()
