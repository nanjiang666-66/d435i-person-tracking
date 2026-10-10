#!/usr/bin/env python3
"""ROS 2 Humble D435i person detection, target tracking and depth output."""

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32, String
from ultralytics import YOLO

from person_vision.depth_estimation import distance_in_box
from person_vision.target_recovery import TargetRecovery
from person_vision.pose_depth import PoseDepthRecovery
from person_vision.segment_depth import SegmentDepthRecovery
from person_vision.timing_diagnostics import FrameTiming


class PersonTracker(Node):
    WINDOW = "D435i person tracking"

    def __init__(self):
        super().__init__("d435i_person_tracker")
        self.declare_parameter("model", "yolo26n.pt")
        self.declare_parameter(
            "tracker", str(Path(__file__).with_name("bytetrack_recovery.yaml"))
        )
        self.declare_parameter("image_size", 416)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("show_window", True)
        self.declare_parameter("display_scale", 2.0)
        self.declare_parameter("target_id", -1)
        self.declare_parameter("auto_reacquire", True)
        self.declare_parameter("reacquire_timeout", 3.0)
        self.declare_parameter("depth_debug", False)
        self.declare_parameter("timing_debug", False)
        self.declare_parameter("pose_depth", False)
        self.declare_parameter("segment_depth", False)

        self.bridge = CvBridge()
        self.model = YOLO(self.get_parameter("model").value)
        self.tracker = str(self.get_parameter("tracker").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.auto_reacquire = bool(self.get_parameter("auto_reacquire").value)
        self.reacquire_timeout = float(self.get_parameter("reacquire_timeout").value)
        self.depth_debug = bool(self.get_parameter("depth_debug").value)
        self.timing_debug = bool(self.get_parameter("timing_debug").value)
        self.frame_timing = FrameTiming()
        self.depth_debug_last_time = 0.0
        self.depth_debug_last_reason = None
        self.pose_depth_enabled = bool(self.get_parameter("pose_depth").value)
        self.segment_depth_enabled = bool(self.get_parameter("segment_depth").value)
        if self.pose_depth_enabled and self.segment_depth_enabled:
            raise ValueError("Select one experimental mode: pose_depth or segment_depth")
        if self.segment_depth_enabled and self.model.task != "segment":
            raise ValueError("segment_depth:=true requires a segmentation model, e.g. yolo26n-seg.pt")
        if self.pose_depth_enabled and self.model.task != "pose":
            raise ValueError("pose_depth:=true requires a pose model, e.g. yolo26n-pose.pt")
        self.pose_recovery = PoseDepthRecovery(max_gap=self.reacquire_timeout)
        self.segment_recovery = SegmentDepthRecovery(max_gap=self.reacquire_timeout)
        if self.image_size <= 0 or self.depth_scale <= 0:
            raise ValueError("image_size and depth_scale must be positive")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.display_scale <= 0.0:
            raise ValueError("display_scale must be positive")
        if self.reacquire_timeout <= 0.0:
            raise ValueError("reacquire_timeout must be positive")
        self.recovery = TargetRecovery(max_gap=self.reacquire_timeout)
        self.boxes = []
        self.confidence_by_id = {}
        self.intrinsics = None
        self.running = True
        self.last_frame_arrival = time.monotonic()
        self.stats_started = time.perf_counter()
        self.stats_color_frames = 0
        self.stats_depth_frames = 0
        self.stats_last_arrival = None
        self.stats_max_gap = 0.0
        self.stats_frames = 0
        self.stats_model_seconds = 0.0
        self.stats_conversion_seconds = 0.0
        self.stats_callback_seconds = 0.0
        self.stats_fps = 0.0
        self.stats_model_ms = 0.0
        self.last_selected_id = -1
        self.target_missing_since = None
        self.depth_jump_active = False

        self.point_pub = self.create_publisher(PointStamped, "/person/target_point", 10)
        self.id_pub = self.create_publisher(Int32, "/person/target_id", 10)
        self.visible_pub = self.create_publisher(Bool, "/person/visible", 10)
        if self.segment_depth_enabled:
            self.detected_pub = self.create_publisher(Bool, "/person/detected", 10)
            self.depth_status_pub = self.create_publisher(String, "/person/depth_status", 10)
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.on_camera_info,
            qos_profile_sensor_data,
        )
        color_sub = Subscriber(
            self,
            Image,
            "/camera/camera/color/image_raw",
            qos_profile=qos_profile_sensor_data,
        )
        depth_sub = Subscriber(
            self,
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            qos_profile=qos_profile_sensor_data,
        )
        color_sub.registerCallback(self.on_color_arrival)
        depth_sub.registerCallback(self.on_depth_arrival)
        self.sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.1
        )
        self.sync.registerCallback(self.on_frame)
        self.color_sub = color_sub
        self.depth_sub = depth_sub
        self.create_timer(0.5, self.check_camera_timeout)
        self.create_timer(5.0, self.report_performance)

        if self.show_window:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.WINDOW, self.on_click)
        self.get_logger().info(
            "YOLO26 CPU tracking ready. "
            + ("Click a person to lock; c clears; q quits."
               if self.show_window else "Headless mode; use target_id to select a person.")
        )

    def check_camera_timeout(self):
        if time.monotonic() - self.last_frame_arrival > 1.0:
            self.visible_pub.publish(Bool(data=False))
            if getattr(self, "segment_depth_enabled", False):
                self.detected_pub.publish(Bool(data=False))
                self.depth_status_pub.publish(String(data="camera_timeout"))

    def report_performance(self):
        now = time.perf_counter()
        elapsed = max(now - self.stats_started, 1e-6)
        self.stats_fps = self.stats_frames / elapsed
        self.stats_model_ms = (
            1000.0 * self.stats_model_seconds / self.stats_frames
            if self.stats_frames else 0.0
        )
        conversion_ms = (
            1000.0 * self.stats_conversion_seconds / self.stats_frames
            if self.stats_frames else 0.0
        )
        callback_ms = (
            1000.0 * self.stats_callback_seconds / self.stats_frames
            if self.stats_frames else 0.0
        )
        self.get_logger().info(
            f"Last {elapsed:.1f}s: processed {self.stats_fps:.1f} FPS, "
            f"color {self.stats_color_frames / elapsed:.1f} FPS, "
            f"depth {self.stats_depth_frames / elapsed:.1f} FPS, "
            f"callback {callback_ms:.0f} ms/frame, "
            f"conversion {conversion_ms:.0f} ms/frame, "
            f"track call {self.stats_model_ms:.0f} ms/frame, "
            f"longest frame gap {self.stats_max_gap:.2f}s, "
            f"last frame {time.monotonic() - self.last_frame_arrival:.2f}s ago"
        )
        if self.timing_debug:
            self.get_logger().info(self.frame_timing.report_and_reset())
        self.stats_started = now
        self.stats_frames = 0
        self.stats_color_frames = 0
        self.stats_depth_frames = 0
        self.stats_model_seconds = 0.0
        self.stats_conversion_seconds = 0.0
        self.stats_callback_seconds = 0.0
        self.stats_max_gap = 0.0

    def on_camera_info(self, message):
        self.intrinsics = (message.k[0], message.k[4], message.k[2], message.k[5])

    def on_color_arrival(self, message):
        self.stats_color_frames += 1
        if self.timing_debug:
            self.frame_timing.record_age(
                "color_rx", self.get_clock().now().nanoseconds * 1e-9, message
            )

    def on_depth_arrival(self, message):
        self.stats_depth_frames += 1
        if self.timing_debug:
            self.frame_timing.record_age(
                "depth_rx", self.get_clock().now().nanoseconds * 1e-9, message
            )

    def on_click(self, event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x = x / self.display_scale
        y = y / self.display_scale
        for track_id, box in self.boxes:
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.set_parameters([Parameter("target_id", value=int(track_id))])
                if getattr(self, "pose_depth_enabled", False):
                    self.pose_recovery.reset()
                if getattr(self, "segment_depth_enabled", False):
                    self.segment_recovery.reset()
                self.get_logger().info(f"Locked target ID {track_id}")
                break

    def change_display_scale(self, delta):
        new_scale = round(min(3.0, max(1.0, self.display_scale + delta)), 2)
        if new_scale == self.display_scale:
            return
        self.display_scale = new_scale
        self.set_parameters([Parameter("display_scale", value=new_scale)])
        self.get_logger().info(f"Display scale: {new_scale:.2f}x")

    def on_frame(self, color_msg, depth_msg):
        segment_enabled = getattr(self, "segment_depth_enabled", False)
        callback_started = time.perf_counter()
        timing_debug = getattr(self, "timing_debug", False)
        if timing_debug:
            ros_now = self.get_clock().now().nanoseconds * 1e-9
            self.frame_timing.record_age("color_entry", ros_now, color_msg)
            self.frame_timing.record_age("depth_entry", ros_now, depth_msg)
            self.frame_timing.record_pair(color_msg, depth_msg)
        arrival = time.monotonic()
        if self.stats_last_arrival is not None:
            self.stats_max_gap = max(
                self.stats_max_gap, arrival - self.stats_last_arrival
            )
        self.stats_last_arrival = arrival
        self.last_frame_arrival = arrival
        conversion_started = time.perf_counter()
        try:
            frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            self.visible_pub.publish(Bool(data=False))
            if segment_enabled:
                self.detected_pub.publish(Bool(data=False))
                self.depth_status_pub.publish(String(data="image_conversion_error"))
            return
        conversion_seconds = time.perf_counter() - conversion_started
        if depth.shape[:2] != frame.shape[:2]:
            self.get_logger().error("Aligned depth and color image sizes differ")
            self.visible_pub.publish(Bool(data=False))
            if segment_enabled:
                self.detected_pub.publish(Bool(data=False))
                self.depth_status_pub.publish(String(data="aligned_image_error"))
            return

        model_started = time.perf_counter()
        try:
            result = self.model.track(
                frame,
                persist=True,
                classes=None if segment_enabled else [0],
                tracker=self.tracker,
                imgsz=self.image_size,
                conf=self.confidence,
                device="cpu",
                verbose=False,
                **({"retina_masks": True} if segment_enabled else {}),
            )[0]
        except Exception as exc:
            self.visible_pub.publish(Bool(data=False))
            if segment_enabled:
                self.detected_pub.publish(Bool(data=False))
                self.depth_status_pub.publish(String(data="inference_error"))
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return
        model_seconds = time.perf_counter() - model_started
        self.boxes = []
        self.confidence_by_id = {}
        masks_by_id = {}
        excluded_mask = None
        if result.boxes is not None and result.boxes.id is not None:
            coordinates = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist() if segment_enabled else [0] * len(ids)
            masks = getattr(result, "masks", None)
            mask_data = masks.data.cpu().numpy() if segment_enabled and masks is not None else None
            for index, (ident, box, confidence, cls) in enumerate(zip(ids, coordinates, confidences, classes)):
                if int(cls) == 0:
                    self.boxes.append((ident, box))
                    self.confidence_by_id[ident] = confidence
                    if mask_data is not None and index < len(mask_data):
                        masks_by_id[ident] = mask_data[index]
                elif (mask_data is not None and index < len(mask_data)
                      and confidence >= .5 and mask_data[index].shape == depth.shape):
                    object_pixels = mask_data[index] > .5
                    excluded_mask = object_pixels if excluded_mask is None else excluded_mask | object_pixels

        selected_id = int(self.get_parameter("target_id").value)
        if selected_id != self.last_selected_id:
            if self.last_selected_id >= 0 and self.target_missing_since is not None:
                missing_seconds = time.monotonic() - self.target_missing_since
                self.get_logger().info(
                    f"Target ID changed from {self.last_selected_id} to {selected_id} "
                    f"after {missing_seconds:.2f}s missing"
                )
            self.target_missing_since = None
            self.last_selected_id = selected_id
            self.recovery.reset()
            if segment_enabled:
                self.segment_recovery.reset()
            self.depth_jump_active = False
        now = time.monotonic()
        selected_box = next(
            (box for track_id, box in self.boxes if track_id == selected_id), None
        )
        selected_present = selected_box is not None
        selected_depth = None
        rejected_depth = None
        body_observation = None
        pose_reason = None
        pose_enabled = getattr(self, "pose_depth_enabled", False)
        structured_depth = pose_enabled or segment_enabled
        if pose_enabled:
            keypoints = getattr(result, "keypoints", None)
            keypoint_data = keypoints.data.cpu().numpy() if keypoints is not None else None
            tracks = [
                {"id": track_id, "box": box, "confidence": self.confidence_by_id[track_id],
                 "keypoints": keypoint_data[index] if keypoint_data is not None else None}
                for index, (track_id, box) in enumerate(self.boxes)
            ]
            stamp = color_msg.header.stamp
            sample_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            old_id = selected_id
            selected_id, body_observation, pose_reason = self.pose_recovery.update(
                frame, depth, depth_msg.encoding, tracks, selected_id, sample_time,
                self.intrinsics[0] if self.intrinsics else 0.0,
                self.depth_scale, self.auto_reacquire,
            )
            if selected_id != old_id:
                self.set_parameters([Parameter("target_id", value=selected_id)])
                self.last_selected_id = selected_id
                self.get_logger().info(
                    f"Auto reacquired target ID {old_id} -> {selected_id}; "
                    "confirmed visible body region and depth"
                )
            selected_present = any(track_id == selected_id for track_id, _ in self.boxes)
            selected_depth = body_observation.depth if body_observation is not None else None
            if self.depth_debug and (
                pose_reason != self.depth_debug_last_reason
                or now - self.depth_debug_last_time >= 1.0
            ):
                selected_track = next(
                    (track for track in tracks if track["id"] == selected_id), None
                )
                confidence_text = "none"
                shoulder_text = hip_text = face_text = "none"
                if selected_track is not None:
                    confidence_text = f"{selected_track['confidence']:.3f}"
                    joints = selected_track["keypoints"]
                    if joints is not None and np.asarray(joints).shape == (17, 3):
                        shoulder_text = ",".join(f"{joints[i, 2]:.3f}" for i in (5, 6))
                        hip_text = ",".join(f"{joints[i, 2]:.3f}" for i in (11, 12))
                        face_text = ",".join(f"{joints[i, 2]:.3f}" for i in (0, 1, 2))
                self.get_logger().info(
                    f"Pose depth: state={pose_reason}; id={selected_id}; "
                    f"depth={selected_depth}; reference={self.pose_recovery.last_depth}; "
                    f"confidence={confidence_text}/0.25(selected); "
                    f"shoulders=[{shoulder_text}]; hips=[{hip_text}]; "
                    f"face=[{face_text}]; "
                    f"reference_region={self.pose_recovery.last_kind}; "
                    f"sample={self.pose_recovery.diagnostic}"
                )
                self.depth_debug_last_reason = pose_reason
                self.depth_debug_last_time = now
        elif segment_enabled:
            tracks = [{"id": ident, "box": box, "confidence": self.confidence_by_id[ident],
                       "mask": masks_by_id.get(ident)} for ident, box in self.boxes]
            stamp = color_msg.header.stamp
            sample_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            old_id = selected_id
            selected_id, body_observation, pose_reason = self.segment_recovery.update(
                frame, depth, depth_msg.encoding, tracks, selected_id, sample_time,
                self.depth_scale, self.auto_reacquire, excluded_mask)
            if selected_id != old_id:
                self.set_parameters([Parameter("target_id", value=selected_id)])
                self.last_selected_id = selected_id
                self.get_logger().info(f"Auto reacquired target ID {old_id} -> {selected_id}; confirmed mask depth")
            selected_present = any(ident == selected_id for ident, _ in self.boxes)
            selected_depth = body_observation.depth if body_observation is not None else None
            if self.depth_debug and (pose_reason != self.depth_debug_last_reason
                                     or now-self.depth_debug_last_time >= 1.):
                self.get_logger().info(
                    f"Segment depth: state={pose_reason}; id={selected_id}; depth={selected_depth}; "
                    f"sample={self.segment_recovery.diagnostic}")
                self.depth_debug_last_reason = pose_reason
                self.depth_debug_last_time = now
        if selected_present and not structured_depth:
            measured_depth = distance_in_box(
                depth, depth_msg.encoding, selected_box, self.depth_scale,
                reference_depth=self.recovery.last_depth,
            )
            if measured_depth is not None:
                if self.recovery.depth_is_plausible(measured_depth, now):
                    selected_depth = measured_depth
                else:
                    rejected_depth = measured_depth
            elif self.recovery.last_depth is not None:
                # The locked range can become stale when the person walks away
                # behind an occluder. Check a well-supported raw torso range,
                # but keep visible=false until the recovery gate confirms it.
                raw_depth = distance_in_box(
                    depth, depth_msg.encoding, selected_box, self.depth_scale,
                    min_roi_support=0.35,
                )
                selected_depth = self.recovery.relock_depth(
                    frame, selected_box, raw_depth,
                    self.confidence_by_id[selected_id], now,
                )
                diagnostic = self.recovery.depth_recovery_diagnostics
                if self.depth_debug and diagnostic is not None and (
                    now - self.depth_debug_last_time >= 1.0
                    or diagnostic["reason"] != self.depth_debug_last_reason
                ):
                    candidate_text = (
                        "none" if raw_depth is None else f"{raw_depth:.3f}m"
                    )
                    self.get_logger().info(
                        f"Depth recovery: reason={diagnostic['reason']}; "
                        f"id={selected_id}; candidate={candidate_text}; "
                        f"reference={diagnostic['reference']:.3f}m; "
                        f"confidence={diagnostic['confidence']:.3f}/0.75; "
                        f"appearance={diagnostic['appearance']:.3f}/0.82; "
                        f"range_limit={diagnostic['range_limit']:.3f}m; "
                        f"stable={diagnostic['stable_frames']}/5"
                    )
                    self.depth_debug_last_time = now
                    self.depth_debug_last_reason = diagnostic["reason"]
                if selected_depth is None:
                    rejected_depth = raw_depth
            self.recovery.record_selected(
                frame, selected_box, selected_depth,
                [(track_id, box, self.confidence_by_id[track_id])
                 for track_id, box in self.boxes if track_id != selected_id],
                now,
            )
        elif selected_id >= 0 and self.auto_reacquire and not structured_depth:
            candidates = []
            for track_id, box in self.boxes:
                confidence = self.confidence_by_id[track_id]
                if confidence < 0.4:
                    continue
                candidate_depth = distance_in_box(
                    depth, depth_msg.encoding, box, self.depth_scale,
                    reference_depth=self.recovery.last_depth,
                )
                if candidate_depth is None and confidence >= 0.75:
                    candidate_depth = distance_in_box(
                        depth, depth_msg.encoding, box, self.depth_scale,
                        min_roi_support=0.35,
                    )
                candidates.append((track_id, box, candidate_depth))
            match = self.recovery.find_match(frame, candidates, now)
            if match is not None:
                new_id, score = match
                missing_seconds = now - self.recovery.last_seen
                old_id = selected_id
                selected_id = new_id
                self.set_parameters([Parameter("target_id", value=new_id)])
                self.last_selected_id = new_id
                self.target_missing_since = None
                selected_box = next(
                    box for track_id, box in self.boxes if track_id == new_id
                )
                selected_depth = next(
                    candidate_depth for track_id, _box, candidate_depth in candidates
                    if track_id == new_id
                )
                selected_present = True
                self.recovery.reset()
                self.recovery.record_selected(
                    frame, selected_box, selected_depth,
                    [(track_id, box, self.confidence_by_id[track_id])
                     for track_id, box in self.boxes if track_id != new_id],
                    now,
                )
                self.get_logger().info(
                    f"Auto reacquired target ID {old_id} -> {new_id} "
                    f"after {missing_seconds:.2f}s; match score {score:.2f}"
                )
        # Either rejection path supplies its actual numeric candidate. The
        # reference-filtered measured_depth may be None during recovery.
        depth_jump = rejected_depth is not None
        if depth_jump and not self.depth_jump_active:
            self.get_logger().warn(
                f"Selected ID {selected_id} depth jump rejected: "
                f"{rejected_depth:.2f}m vs previous {self.recovery.last_depth:.2f}m"
            )
        self.depth_jump_active = depth_jump
        if selected_id >= 0 and not selected_present:
            if self.target_missing_since is None:
                self.target_missing_since = time.monotonic()
                detected_ids = [track_id for track_id, _ in self.boxes]
                self.get_logger().warn(
                    f"Selected ID {selected_id} missing; visible=false; "
                    f"detected IDs: {detected_ids}"
                )
        elif selected_id >= 0 and self.target_missing_since is not None:
            missing_seconds = time.monotonic() - self.target_missing_since
            self.get_logger().info(
                f"Selected ID {selected_id} restored after {missing_seconds:.2f}s"
            )
            self.target_missing_since = None
        visible = False
        if segment_enabled and self.show_window and self.depth_debug:
            sample = self.segment_recovery.sample_pixels
            if sample is not None:
                sampled_frame = frame[::2, ::2]
                sampled_frame[sample] = (.65*sampled_frame[sample] + .35*np.array([0, 255, 0])).astype(np.uint8)
        if pose_enabled and self.show_window and self.depth_debug:
            sample = self.pose_recovery.diagnostic
            if 'roi' in sample:
                sx1, sy1, sx2, sy2 = sample['roi']
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 255), 1)
                cv2.putText(frame, f"depth ROI: {sample['region']}",
                            (sx1, max(20, sy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 0, 255), 1)
        for track_id, box in self.boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            selected = track_id == selected_id
            color = (0, 255, 0) if selected else (255, 160, 0)
            label = f"person ID {track_id} {self.confidence_by_id[track_id]:.2f}"

            if selected:
                distance = selected_depth
                if distance is not None and self.intrinsics is not None:
                    fx, fy, cx, cy = self.intrinsics
                    if fx > 0 and fy > 0 and np.isfinite([fx, fy, cx, cy]).all():
                        u, v = (
                            (body_observation.u, body_observation.v)
                            if body_observation is not None
                            else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        )
                        point = PointStamped()
                        point.header = color_msg.header  # camera optical frame
                        point.point.x = (u - cx) * distance / fx
                        point.point.y = (v - cy) * distance / fy
                        point.point.z = distance
                        if timing_debug:
                            self.frame_timing.record_age(
                                "xyz_pub", self.get_clock().now().nanoseconds * 1e-9, point
                            )
                            self.frame_timing.record_ms(
                                "compute_to_pub", (time.perf_counter() - callback_started) * 1000.0
                            )
                        self.point_pub.publish(point)
                        self.id_pub.publish(Int32(data=track_id))
                        label += f"  {distance:.2f} m"
                        if body_observation is not None:
                            label += f" ({body_observation.kind})"
                        if body_observation is not None and self.show_window:
                            cv2.circle(frame, (round(u), round(v)), 4, (0, 255, 255), -1)
                        visible = True
                else:
                    if pose_reason in ("confirming_body_depth", "confirming_mask_depth"):
                        label += "  confirming depth"
                    elif pose_reason == "uncertain_pose":
                        label += "  no reliable depth (pose)"
                    elif pose_reason == "low_detection_confidence":
                        label += "  no reliable depth (confidence)"
                    else:
                        label += "  depth jump" if depth_jump else "  no reliable depth"
            if self.show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

        self.visible_pub.publish(Bool(data=visible))
        if segment_enabled:
            self.detected_pub.publish(Bool(data=selected_present))
            status = pose_reason if not selected_depth or visible else "no_intrinsics"
            self.depth_status_pub.publish(String(data=status))
        if self.show_window:
            cv2.putText(frame, f"5s avg: {self.stats_fps:.1f} FPS", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if self.show_window:
            if selected_id < 0:
                status = "Click a person"
            elif selected_present:
                status = f"Target: {selected_id}"
            else:
                status = f"Target: {selected_id} LOST"
            hint = (
                " | click person | c: clear | q: quit"
                if selected_id >= 0 and not selected_present
                else " | c: clear | q: quit"
            )
            cv2.putText(frame, status + hint, (12, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Scale {self.display_scale:.2f}x | +/-: resize image",
                        (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            display_width = round(frame.shape[1] * self.display_scale)
            display_height = round(frame.shape[0] * self.display_scale)
            display_frame = cv2.resize(
                frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR
            )
            cv2.imshow(self.WINDOW, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                self.set_parameters([Parameter("target_id", value=-1)])
            elif key == ord("q"):
                self.running = False
            elif key in (ord("+"), ord("=")):
                self.change_display_scale(0.25)
            elif key in (ord("-"), ord("_")):
                self.change_display_scale(-0.25)
        self.stats_frames += 1
        self.stats_model_seconds += model_seconds
        self.stats_conversion_seconds += conversion_seconds
        self.stats_callback_seconds += time.perf_counter() - callback_started


def main():
    rclpy.init()
    node = None
    try:
        node = PersonTracker()
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
            if node.show_window:
                cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
