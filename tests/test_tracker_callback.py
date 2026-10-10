"""Run the production frame callback with synthetic RGB-D and fake ROS I/O.

This covers processing and publication, including warning formatting. It does
not exercise a ROS executor, real YOLO inference, or camera hardware.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def int(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


def load_tracker_module():
    """Stub only external adapters; use the real callback and depth/recovery code."""
    def point_stamped():
        return SimpleNamespace(header=None, point=SimpleNamespace(x=0, y=0, z=0))

    adapters = {
        "cv2": {"waitKey": Mock(return_value=-1), "resize": Mock()},
        "rclpy": {},
        "cv_bridge": {"CvBridge": Mock()},
        "geometry_msgs.msg": {"PointStamped": point_stamped},
        "message_filters": {"ApproximateTimeSynchronizer": Mock(), "Subscriber": Mock()},
        "rclpy.node": {"Node": object},
        "rclpy.parameter": {"Parameter": lambda name, value: SimpleNamespace(name=name, value=value)},
        "rclpy.qos": {"qos_profile_sensor_data": object()},
        "sensor_msgs.msg": {"CameraInfo": object, "Image": object},
        "std_msgs.msg": {"Bool": SimpleNamespace, "Int32": SimpleNamespace, "String": SimpleNamespace},
        "ultralytics": {"YOLO": Mock()},
    }
    stubs = {}
    for name, attributes in adapters.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        stubs[name] = module
    cv = stubs["cv2"]
    cv.rectangle = Mock()
    cv.circle = Mock()
    cv.putText = Mock()
    cv.imshow = Mock()
    cv.FONT_HERSHEY_SIMPLEX = 0
    cv.INTER_LINEAR = 1
    path = Path(__file__).resolve().parents[1] / "person_vision" / "tracker_node.py"
    spec = importlib.util.spec_from_file_location("_tracker_callback_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class TrackerCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_tracker_module()

    def setUp(self):
        self.now = 100.0
        self.clock_patch = patch.object(self.module, "time", SimpleNamespace(
            monotonic=lambda: self.now, perf_counter=lambda: self.now,
        ))
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.target_id = 7
        self.logger = Mock()
        self.frame = np.full((100, 100, 3), 220, dtype=np.uint8)
        self.node = SimpleNamespace(
            WINDOW="test", bridge=Mock(), model=Mock(),
            tracker="bytetrack_recovery.yaml", image_size=416, confidence=0.25,
            depth_scale=0.001, show_window=False, display_scale=1.0,
            auto_reacquire=True, recovery=self.module.TargetRecovery(),
            depth_debug=False, depth_debug_last_time=0.0, depth_debug_last_reason=None,
            last_selected_id=7, target_missing_since=None, depth_jump_active=False,
            intrinsics=(100.0, 100.0, 50.0, 50.0),
            stats_last_arrival=None, stats_max_gap=0.0, stats_frames=0,
            stats_model_seconds=0.0, stats_conversion_seconds=0.0,
            stats_callback_seconds=0.0, stats_fps=20.0,
            point_pub=Mock(), visible_pub=Mock(), id_pub=Mock(),
            get_parameter=lambda _name: SimpleNamespace(value=self.target_id),
            get_logger=lambda: self.logger, set_parameters=self.set_parameters,
        )
        self.node.bridge.imgmsg_to_cv2.side_effect = lambda message, **kwargs: message.pixels
        self.process(1000)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.node.point_pub.reset_mock()
        self.logger.reset_mock()

    def set_parameters(self, parameters):
        self.target_id = parameters[0].value

    def process(self, millimetres, track_id=7, elapsed=0.05):
        self.now += elapsed
        boxes = None if track_id is None else SimpleNamespace(
            xyxy=Tensor([[0, 0, 100, 100]]), id=Tensor([track_id]), conf=Tensor([0.9]),
        )
        self.node.model.track.return_value = [SimpleNamespace(boxes=boxes)]
        stamp_time = self.now - 0.30
        self.last_color_header = SimpleNamespace(
            frame_id="camera_color_optical_frame",
            stamp=SimpleNamespace(sec=int(stamp_time), nanosec=round((stamp_time-int(stamp_time))*1e9)),
        )
        color = SimpleNamespace(pixels=self.frame.copy(), header=self.last_color_header)
        depth = SimpleNamespace(
            pixels=np.full((100, 100), millimetres, dtype=np.uint16), encoding="16UC1",
            header=self.last_color_header,
        )
        self.module.PersonTracker.on_frame(self.node, color, depth)

    def assert_invalid_without_position(self):
        self.assertFalse(self.node.visible_pub.publish.call_args.args[0].data)
        self.node.point_pub.publish.assert_not_called()

    def enable_timing(self):
        self.node.timing_debug = True
        self.node.frame_timing = self.module.FrameTiming()
        self.node.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=round(self.now*1e9))
        )

    def test_timing_diagnostic_preserves_original_measurement_header_and_position(self):
        self.enable_timing()
        self.process(1000)
        published = self.node.point_pub.publish.call_args.args[0]
        self.assertIs(published.header, self.last_color_header)
        self.assertAlmostEqual(published.point.z, 1.0)
        self.assertEqual(published.header.frame_id, "camera_color_optical_frame")
        self.assertAlmostEqual(self.node.frame_timing.samples["xyz_pub"][-1], 300.0)

    def test_timing_diagnostic_does_not_publish_position_for_invalid_depth(self):
        self.enable_timing()
        self.process(0)
        self.assert_invalid_without_position()
        self.assertEqual(len(self.node.frame_timing.samples["xyz_pub"]), 0)
        self.assertEqual(len(self.node.frame_timing.samples["color_entry"]), 1)

    def test_occluder_warning_does_not_crash_and_clear_frame_recovers(self):
        self.node.show_window = True
        for _ in range(8):
            self.process(550)
            self.assert_invalid_without_position()
        self.logger.warn.assert_called_once()
        self.assertIn("0.55m", self.logger.warn.call_args.args[0])
        self.assertAlmostEqual(self.node.recovery.last_depth, 1.0)
        self.process(1000)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertAlmostEqual(self.node.point_pub.publish.call_args.args[0].point.z, 1.0)

    def test_retreat_waits_for_confirmation_then_publishes_new_range(self):
        self.node.depth_debug = True
        self.process(1450, elapsed=1.0)
        self.assert_invalid_without_position()
        for _ in range(3):
            self.process(1450)
            self.assert_invalid_without_position()
        self.assertIn("1.45m", self.logger.warn.call_args.args[0])
        self.process(1450)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertAlmostEqual(self.node.point_pub.publish.call_args.args[0].point.z, 1.45, places=3)
        self.assertIn("reason=accepted", self.logger.info.call_args.args[0])

    def test_missing_depth_publishes_invalid_without_warning_format_error(self):
        self.node.depth_debug = True
        self.process(0)
        self.assert_invalid_without_position()
        self.logger.warn.assert_not_called()
        self.assertIn("candidate=none", self.logger.info.call_args.args[0])
        self.assertIn("reason=no_supported_depth", self.logger.info.call_args.args[0])

    def test_diagnostic_identifies_appearance_gate_and_throttles_repeats(self):
        self.node.depth_debug = True
        self.frame[:] = 30
        for _ in range(8):
            self.process(1450)
            self.assert_invalid_without_position()
        self.logger.info.assert_called_once()
        report = self.logger.info.call_args.args[0]
        self.assertIn("reason=appearance_mismatch", report)
        self.assertIn("candidate=1.450m", report)
        self.assertIn("reference=1.000m", report)
        self.assertIn("appearance=0.000/0.82", report)

    def test_fast_small_jump_uses_measured_range_in_warning(self):
        self.process(1250, elapsed=0.01)
        self.assert_invalid_without_position()
        self.assertIn("1.25m", self.logger.warn.call_args.args[0])

    def test_new_id_after_occlusion_can_recover_farther_range(self):
        self.process(0, track_id=None, elapsed=0.5)
        self.assert_invalid_without_position()
        for _ in range(4):
            self.process(1450, track_id=8)
            self.assert_invalid_without_position()
        self.process(1450, track_id=8)
        self.assertEqual(self.target_id, 8)
        self.assertTrue(self.node.visible_pub.publish.call_args.args[0].data)
        self.assertAlmostEqual(self.node.point_pub.publish.call_args.args[0].point.z, 1.45, places=3)


if __name__ == "__main__":
    unittest.main()
