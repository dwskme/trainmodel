#!/usr/bin/env python3
import math
import os

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from model.unet import UNet
from postprocess import process_prediction
from transforms import apply_filters


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__("lane_follower")

        # Parameters
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("model_path", "lane_unet.pth")
        self.declare_parameter("forward_speed", 0.2)  # m/s
        self.declare_parameter("steer_gain", 0.01)  # rad per degree
        self.declare_parameter("max_steer", 0.5)  # rad/sec max

        # Load params
        cam_topic = self.get_parameter("camera_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value
        model_path = self.get_parameter("model_path").value
        self.v = self.get_parameter("forward_speed").value
        self.k = self.get_parameter("steer_gain").value
        self.max_w = self.get_parameter("max_steer").value

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.get_logger().info("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            self.get_logger().info("Using CPU device")

        # Load model
        self.model = UNet().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

        # ROS interfaces
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, cam_topic, self.image_callback, 1
        )
        # optional odom subscriber
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )
        self.pub_twist = self.create_publisher(Twist, cmd_topic, 1)

        # state
        self.current_yaw = 0.0

        self.get_logger().info("Lane follower node started.")

    def odom_callback(self, msg: Odometry):
        # Stub: extract yaw if you want heading fusion
        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def image_callback(self, msg: Image):
        # Convert ROS Image → CV2 BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h_full, w_full = frame.shape[:2]

        # Preprocess & inference
        small = cv2.resize(frame, (256, 256))
        filt = apply_filters(small)

        # to float32 tensor
        np_img = filt.astype("float32") / 255.0
        tensor = (
            torch.from_numpy(np_img.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            clean, lines, angle_deg = process_prediction(self.model(tensor))

        # Compute steering command
        # Convert degrees → radians, apply gain
        steer = -angle_deg * (math.pi / 180.0) * self.k
        steer = max(-self.max_w, min(self.max_w, steer))

        # Publish Twist
        twist = Twist()
        twist.linear.x = float(self.v)
        twist.angular.z = float(steer)
        self.pub_twist.publish(twist)

        # (Optional) debug log
        self.get_logger().debug(f"angle={angle_deg:.1f}°, steer={steer:.3f}")

    def destroy_node(self):
        super().destroy_node()
        self.get_logger().info("Lane follower node shutting down.")


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
