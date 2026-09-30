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
from std_msgs.msg import Bool, Int32
from ultralytics import YOLO

from person_vision.target_recovery import TargetRecovery


def distance_in_box(depth, encoding, box, depth_scale):
    """Median valid torso-region depth in metres; None if no usable pixels."""
    x1, y1, x2, y2 = (int(v) for v in box)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None

    # The centre of a person box tends to contain fewer background pixels.
    left = max(0, x1 + int(width * 0.3))
    right = min(depth.shape[1], x1 + int(width * 0.7))
    top = max(0, y1 + int(height * 0.2))
    bottom = min(depth.shape[0], y1 + int(height * 0.65))
    if left >= right or top >= bottom:
        return None

    pixels = depth[top:bottom, left:right].astype(np.float32)
    if encoding == "16UC1":
        pixels *= depth_scale
    elif encoding != "32FC1":
        return None
    valid = pixels[np.isfinite(pixels) & (pixels >= 0.2) & (pixels <= 8.0)]
    return float(np.median(valid)) if valid.size >= 20 else None


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

    def on_color_arrival(self, _message):
        self.stats_color_frames += 1

    def on_depth_arrival(self, _message):
        self.stats_depth_frames += 1

    def on_click(self, event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x = x / self.display_scale
        y = y / self.display_scale
        for track_id, box in self.boxes:
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.set_parameters([Parameter("target_id", value=int(track_id))])
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
        callback_started = time.perf_counter()
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
            return
        conversion_seconds = time.perf_counter() - conversion_started
        if depth.shape[:2] != frame.shape[:2]:
            self.get_logger().error("Aligned depth and color image sizes differ")
            self.visible_pub.publish(Bool(data=False))
            return

        model_started = time.perf_counter()
        try:
            result = self.model.track(
                frame,
                persist=True,
                classes=[0],  # COCO class 0: person
                tracker=self.tracker,
                imgsz=self.image_size,
                conf=self.confidence,
                device="cpu",
                verbose=False,
            )[0]
        except Exception as exc:
            self.visible_pub.publish(Bool(data=False))
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return
        model_seconds = time.perf_counter() - model_started
        self.boxes = []
        self.confidence_by_id = {}
        if result.boxes is not None and result.boxes.id is not None:
            coordinates = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            self.boxes = list(zip(ids, coordinates))
            self.confidence_by_id = dict(zip(ids, confidences))

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
            self.depth_jump_active = False
        now = time.monotonic()
        selected_box = next(
            (box for track_id, box in self.boxes if track_id == selected_id), None
        )
        selected_present = selected_box is not None
        selected_depth = None
        depth_jump = False
        if selected_present:
            measured_depth = distance_in_box(
                depth, depth_msg.encoding, selected_box, self.depth_scale
            )
            if measured_depth is not None:
                if self.recovery.depth_is_plausible(measured_depth, now):
                    selected_depth = measured_depth
                else:
                    depth_jump = True
            self.recovery.record_selected(
                frame, selected_box, selected_depth,
                [track_id for track_id, _ in self.boxes if track_id != selected_id],
                now,
            )
        elif selected_id >= 0 and self.auto_reacquire:
            candidates = [
                (track_id, box, distance_in_box(
                    depth, depth_msg.encoding, box, self.depth_scale
                ))
                for track_id, box in self.boxes
                if self.confidence_by_id[track_id] >= 0.4
            ]
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
                    [track_id for track_id, _ in self.boxes if track_id != new_id],
                    now,
                )
                self.get_logger().info(
                    f"Auto reacquired target ID {old_id} -> {new_id} "
                    f"after {missing_seconds:.2f}s; match score {score:.2f}"
                )
        if depth_jump and not self.depth_jump_active:
            self.get_logger().warn(
                f"Selected ID {selected_id} depth jump rejected: "
                f"{measured_depth:.2f}m vs previous {self.recovery.last_depth:.2f}m"
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
        for track_id, box in self.boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            selected = track_id == selected_id
            color = (0, 255, 0) if selected else (255, 160, 0)
            label = f"person ID {track_id} {self.confidence_by_id[track_id]:.2f}"

            if selected:
                distance = selected_depth
                if distance is not None and self.intrinsics is not None:
                    fx, fy, cx, cy = self.intrinsics
                    if fx > 0 and fy > 0:
                        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        point = PointStamped()
                        point.header = color_msg.header  # camera optical frame
                        point.point.x = (u - cx) * distance / fx
                        point.point.y = (v - cy) * distance / fy
                        point.point.z = distance
                        self.point_pub.publish(point)
                        self.id_pub.publish(Int32(data=track_id))
                        label += f"  {distance:.2f} m"
                        visible = True
                else:
                    label += "  depth jump" if depth_jump else "  no valid depth"
            if self.show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

        self.visible_pub.publish(Bool(data=visible))
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
