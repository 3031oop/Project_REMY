import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ultralytics import YOLO
from turtlebot3_msgs.srv import Sound as SoundSrv
from std_msgs.msg import String

import pyrealsense2 as rs
import numpy as np
import time
import math
import cv2
import threading
from pathlib import Path

class FallingObjectSmartReturn(Node):
    def __init__(self):
        super().__init__('yolo_state_node')
        
        # 1) YOLO Model Setup
        model_path = Path('/home/remi/kitchen/kitchen_ncnn_model2')
        self.model = YOLO(str(model_path), task='detect')
        self.bridge = CvBridge()

        # 2) RealSense Setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.width, self.height = 640, 480
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 15)
        profile = self.pipeline.start(config)

        #realsense ready time
        time.sleep(1.0)
        # Laser power & Visual Preset for depth quality
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 2) # High Density preset
        
        # Post-processing filters for noise reduction
        self.spatial = rs.spatial_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.align = rs.align(rs.stream.color) 

        # 3) State & Control Variables
        self.state = 'IDLE'
        self.prev_sent_state = None
        self.current_odom = None
        self.latest_target = None
        self.lock = threading.Lock()

        # Navigation & Pose memory
        self.start_pose = None
        self.start_orientation = 0.0
        self.move_start_pose = None
        self.target_dist = 0.0
        
        # Odometry-based Patrol Variables (Angles in Radians)
        self.patrol_sub_state = 'MOVE'
        self.patrol_start_pose = None
        self.last_yaw = 0.0
        self.patrol_rotated_yaw = 0.0  # Accumulated rotation angle
        
        # Detection Hysteresis & Precision Control
        self.stop_count = 0 
        self.last_error = 0.0

        # 4) ROS Publisher/Subscriber
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.image_out_pub = self.create_publisher(Image, '/image_out', 10)
        self.command_pub = self.create_publisher(String, '/waffle_command', 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.command_sub = self.create_subscription(String, '/waffle_command', self.command_callback, 10)
        self.sound_client = self.create_client(SoundSrv, '/sound')

        # 5) Start Background Threads & Control Timer
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
        self.timer = self.create_timer(0.1, self.main_control_loop) # 10Hz

    def odom_callback(self, msg):
        self.current_odom = msg

    def get_yaw_from_quaternion(self, q):
        """Convert quaternion orientation to yaw (euler angle)."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_current_xy(self):
        """Extract x, y coordinates from current odometry."""
        p = self.current_odom.pose.pose.position
        return float(p.x), float(p.y)

    def inference_loop(self):
        """Background thread for YOLO inference to maintain high FPS."""
        time.sleep(2.0)
        while rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if not frames: continue
                
                aligned_frames = self.align.process(frames)
                depth_frame = self.hole_filling.process(self.spatial.process(aligned_frames.get_depth_frame())).as_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame: continue

                color_image = np.asanyarray(color_frame.get_data())
                
                # Predict objects using YOLO (NCNN/TensorRT recommended for speed)
                results = self.model.predict(source=color_image, conf=0.6, imgsz=320, device='cpu', verbose=False)
                
                best_det = {'cx': None, 'dist': 0.0, 'img': color_image.copy()}
                if results and len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                    cx, cy = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                    
                    if 0 <= cx < self.width and 0 <= cy < self.height:
                        dist = float(depth_frame.get_distance(cx, cy))
                        # Validate depth range (0.2m ~ 4.0m)
                        if 0.2 < dist < 4.0:
                            cv2.rectangle(best_det['img'], (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                            best_det.update({'cx': cx, 'dist': dist})

                with self.lock:
                    self.latest_target = best_det
            except Exception as e:
                self.get_logger().warn(f"Inference Thread Error: {e}")
            time.sleep(0.05)

    def main_control_loop(self):
        """Main control logic executed at 10Hz."""
        if self.current_odom is None: return
        with self.lock:
            target = self.latest_target
        if target is None: return
        
        display_image = target['img']
        cur_yaw = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)

        # --- FSM (Finite State Machine) Logic ---
        if self.state == 'IDLE':
            self.start_pose = self.get_current_xy()
            self.start_orientation = cur_yaw
            self.stop_count = 0

        elif self.state == 'PATROL':
            self.publish_msgs("patrol")

            if target['cx'] is not None:
                # 1. Target glimpsed: Slow down for precision scan
                self.send_control(0.0, 0.08) 
                self.stop_count += 1
                self.last_yaw = cur_yaw # Sync yaw to prevent accumulation while waiting
                if self.stop_count >= 5: # Confirm target (stable detection)
                    self.send_control(0.0, 0.0)
                    self.state = 'DETECT'
                    self.stop_count = 0
                return
            else:
                # 2. Target lost: Wait shortly (Hysteresis) before resuming rotation
                if self.stop_count > 0:
                    self.stop_count -= 1
                    self.send_control(0.0, 0.0)
                    self.last_yaw = cur_yaw
                    return

            # 3. No target: Resume patrol sub-logic
            cur_x, cur_y = self.get_current_xy()
            if self.patrol_sub_state == 'MOVE':
                if self.patrol_start_pose is None: self.patrol_start_pose = (cur_x, cur_y)
                dist = math.sqrt((cur_x - self.patrol_start_pose[0])**2 + (cur_y - self.patrol_start_pose[1])**2)
                if dist < 1.0: self.send_control(0.08, 0.0) # Move forward 1m
                else:
                    self.send_control(0.0, 0.0)
                    self.patrol_sub_state = 'ROTATE'
                    self.patrol_rotated_yaw = 0.0
                    self.last_yaw = cur_yaw

            elif self.patrol_sub_state == 'ROTATE':
                # Accumulate actual rotation angle using Odom
                diff = cur_yaw - self.last_yaw
                if diff > math.pi: diff -= 2 * math.pi
                if diff < -math.pi: diff += 2 * math.pi
                self.patrol_rotated_yaw += abs(diff)
                self.last_yaw = cur_yaw

                if self.patrol_rotated_yaw < 2 * math.pi: # Complete full 360-degree turn
                    self.send_control(0.0, 0.25)
                else:
                    self.get_logger().warn("Patrol Finished. Heading back home.")
                    self.state = 'RETURN'

        elif self.state == 'DETECT':
            self.publish_msgs("detect")
            if target['cx'] is not None:
                self.target_dist = target['dist']
                self.state = 'ROTATE_TO_TARGET'
            else:
                self.state = 'PATROL'

        elif self.state == 'ROTATE_TO_TARGET':
            if target['cx'] is not None:
                error = (self.width / 2) - target['cx']
                if abs(error) > 20: # Center the target in the frame
                    angular_z = max(-0.2, min(0.2, error * 0.0015))
                    self.send_control(0.0, angular_z)
                else:
                    self.send_control(0.0, 0.0)
                    self.move_start_pose = self.get_current_xy()
                    self.state = 'MOVING'
            else: self.state = 'DETECT'

        elif self.state == 'MOVING':
            cur_x, cur_y = self.get_current_xy()
            moved = math.sqrt((cur_x - self.move_start_pose[0])**2 + (cur_y - self.move_start_pose[1])**2)
            # Stop 0.4m before the target
            if moved < max(0.0, self.target_dist - 0.4):
                self.send_control(0.12, 0.0)
            else:
                self.send_control(0.0, 0.0)
                self.publish_msgs("depart")
                self.wait_start_time = time.time()
                self.state = 'WAIT'

        elif self.state == 'WAIT':
            self.publish_msgs("wait_return")
            if time.time() - self.wait_start_time > 10.0: # Automatic return after 10s
                self.publish_msgs("force_return")
                self.state = 'RETURN'

        elif self.state == 'RETURN':
            self.publish_msgs("return")
            cur_x, cur_y = self.get_current_xy()
            dx, dy = self.start_pose[0] - cur_x, self.start_pose[1] - cur_y
            dist_to_home = math.sqrt(dx**2 + dy**2)
            angle_to_home = math.atan2(dy, dx)
            angle_diff = (angle_to_home - cur_yaw + math.pi) % (2 * math.pi) - math.pi

            if dist_to_home > 0.08:
                if abs(angle_diff) > 0.15: # Face home direction first
                    self.send_control(0.0, 0.4 if angle_diff > 0 else -0.4)
                else: # Drive straight
                    self.send_control(0.15, 0.0)
            else: self.state = 'ALIGN_FINAL'

        elif self.state == 'ALIGN_FINAL':
            # Align orientation to match original start orientation
            diff = (self.start_orientation - cur_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > 0.05: self.send_control(0.0, 0.3 if diff > 0 else -0.3)
            else:
                self.send_control(0.0, 0.0)
                self.get_logger().info("Mission Complete. IDLE.")
                self.state = 'IDLE'
                self.prev_sent_state = None

        # --- UI Overlay & Pub ---
        cv2.putText(display_image, f"STATE: {self.state}", (20, 40), 1, 1.5, (255,0,0), 2)
        if self.state == 'PATROL':
            cv2.putText(display_image, f"Rot: {math.degrees(self.patrol_rotated_yaw):.1f}deg", (20, 70), 1, 1.5, (0,255,255), 2)
        self.image_out_pub.publish(self.bridge.cv2_to_imgmsg(display_image, encoding='bgr8'))

    def send_control(self, linear, angular):
        """Publish velocity command."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

    def publish_msgs(self, status):
        """Publish system status messages if changed."""
        if self.prev_sent_state != status:
            msg = String(); msg.data = status
            self.command_pub.publish(msg)
            self.prev_sent_state = status

    def command_callback(self, msg):
        """Handle incoming command messages."""
        cmd = msg.data.lower().strip()
        if cmd == 'start' and self.state == 'IDLE':
            self.state = 'PATROL'
            self.patrol_sub_state = 'MOVE'
            self.patrol_start_pose = None
        elif cmd == 'return':
            self.state = 'RETURN'

def main():
    rclpy.init()
    node = FallingObjectSmartReturn()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
