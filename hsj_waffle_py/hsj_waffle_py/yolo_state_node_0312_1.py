import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from turtlebot3_msgs.msg import Sound
from cv_bridge import CvBridge
from ultralytics import YOLO

import pyrealsense2 as rs
import numpy as np
import time
import math
import cv2
from pathlib import Path

class FallingObjectSmartReturn(Node):
    def __init__(self):
        super().__init__('yolo_state_node')

        # 1) NCNN model load 
        self.model = None
        model_path = Path('/home/remi/kitchen/kitchen_ncnn_model')
        try:
            self.model = YOLO(str(model_path), task='detect')
            self.get_logger().info(f'NCNN model loaded: {model_path}')
        except Exception as e:
            self.get_logger().error(f'NCNN load error: {e}')

        self.bridge = CvBridge()

        # 2) RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.width, self.height = 640, 480
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 15)
        self.pipeline.start(config)
        #self.align = rs.align(rs.stream.color)

        # state setup
        self.state = 'IDLE'
        self.current_odom = None

        # start inform (for return)
        self.start_pose = None           # (x, y)
        self.start_orientation = 0.0     # start Yaw(angle)

        self.move_start_pose = None      #  pose at detect 
        self.stop_count = 0
        self.prev_dist = 0.0

        self.target_dist = 0.0
        self.target_angle = 0.0
        self.rotate_cmd = 0.0
        self.rotate_end_time = None
        self.wait_start_time = None

        # ROS pub/sub
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.sound_pub = self.create_publisher(Sound, '/sound', 10)
        self.image_out_pub = self.create_publisher(Image, '/image_out', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.timer = self.create_timer(0.1, self.process_loop)

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.current_odom = msg

    def get_current_xy(self):
        p = self.current_odom.pose.pose.position
        return float(p.x), float(p.y)

    def detect_best_object(self, color_image, depth_frame, display_image):
        if self.model is None: return None
        results = self.model.predict(source=color_image, conf=0.5, imgsz=320, device='cpu', verbose=False)
        if not results or len(results[0].boxes) == 0: return None

        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        best = None
        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.width-1, x2), min(self.height-1, y2)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            dist = float(depth_frame.get_distance(cx, cy))

            if 0.3 < dist < 3.0:
                if best is None or conf > best['conf']:
                    best = {'cx': cx, 'dist': dist, 'conf': float(conf), 'cls_name': self.model.names[cls_id]}
            
            # image
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, f"{dist:.2f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return best

    def process_loop(self):
        try:
            frames = self.pipeline.wait_for_frames()
            #aligned_frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame or self.current_odom is None: return

            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()

            # --- State Machine ---
            
            # [1] IDLE
            if self.state == 'IDLE':
                self.start_pose = self.get_current_xy()
                self.start_orientation = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)
                self.stop_count = 0
                self.get_logger().info(f'Step 1: Start Pose Saved. Yaw: {math.degrees(self.start_orientation):.2f}deg')
                self.state = 'DETECT'

            # [2] DETECT
            elif self.state == 'DETECT':
                target = self.detect_best_object(color_image, depth_frame, display_image)
                if target:
                    if abs(self.prev_dist - target['dist']) < 0.01: self.stop_count += 1
                    else: self.stop_count = 0
                    self.prev_dist = target['dist']

                    if self.stop_count > 10:
                        self.target_dist = target['dist']
                        self.target_angle = -((target['cx'] - (self.width / 2)) / float(self.width)) * 1.2
                        self.rotate_cmd = 0.4 if self.target_angle > 0 else -0.4
                        self.rotate_end_time = time.time() + abs(self.target_angle) * 1.3
                        self.trigger_sound()
                        self.state = 'ROTATE'

            # [3] ROTATE
            elif self.state == 'ROTATE':
                if self.rotate_end_time and time.time() < self.rotate_end_time:
                    self.send_control(0.0, self.rotate_cmd)
                else:
                    self.send_control(0.0, 0.0)
                    self.move_start_pose = self.get_current_xy()
                    self.state = 'MOVING'

            # [4] MOVING
            elif self.state == 'MOVING':
                cur_x, cur_y = self.get_current_xy()
                moved = math.sqrt((cur_x - self.move_start_pose[0])**2 + (cur_y - self.move_start_pose[1])**2)
                if moved < max(0.0, self.target_dist - 0.4):
                    self.send_control(0.1, 0.0)
                else:
                    self.send_control(0.0, 0.0)
                    self.wait_start_time = time.time()
                    self.state = 'WAIT'

            # [5] WAIT
            elif self.state == 'WAIT':
                if time.time() - self.wait_start_time > 2.0:
                    self.state = 'RETURN'

            # [6] RETURN
            elif self.state == 'RETURN':
                cur_x, cur_y = self.get_current_xy()
                remain = math.sqrt((self.start_pose[0] - cur_x)**2 + (self.start_pose[1] - cur_y)**2)
                if remain > 0.05:
                    self.send_control(-0.1, 0.0)
                else:
                    self.send_control(0.0, 0.0)
                    self.get_logger().info('Position restored. Aligning orientation...')
                    self.state = 'ALIGN_FINAL'

            # [7] ALIGN_FINAL:(YAW)
            elif self.state == 'ALIGN_FINAL':
                cur_yaw = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)
                diff = self.start_orientation - cur_yaw
                # regularization normalization (-pi to pi)
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                
                if abs(diff) > 0.05: #  3 angle diff
                    self.send_control(0.0, 0.3 if diff > 0 else -0.3)
                else:
                    self.send_control(0.0, 0.0)
                    self.get_logger().info('Step 7: Perfect Home Return! Mission Complete.')
                    self.state = 'IDLE'

            # state image
            cv2.putText(display_image, f"STATE: {self.state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            self.image_out_pub.publish(self.bridge.cv2_to_imgmsg(display_image, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f"Loop Error: {str(e)}")

    def trigger_sound(self):
        s_msg = Sound()
        s_msg.value = Sound.ERROR
        self.sound_pub.publish(s_msg)

    def send_control(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

def main():
    rclpy.init()
    node = FallingObjectSmartReturn()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
