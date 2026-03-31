import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from turtlebot3_msgs.msg import Sound
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import time
import math
import os

class KitchenToolsFinalNode(Node):
    def __init__(self):
        super().__init__('kitchen_tools_final_node')
        
        #home_dir = os.path.expanduser('~')
        #model_path = os.path.join(home_dir, 'kitchen', 'kitchen_tools_ncnn')
        
        self.model = YOLO('/home/remi/kitchen', task='detect')
        self.bridge = CvBridge()
        self.state = 'IDLE'
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sound_pub = self.create_publisher(Sound, '/sound', 10)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.latest_depth = None
        self.current_odom = None
        self.idle_start_time = time.time()
        self.start_pose = None
        self.start_orientation = 0.0
        self.target_angle = 0.0
        self.target_dist = 0.0
        self.prev_dist = 0.0
        self.stop_count = 0

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.current_odom = msg

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def image_callback(self, msg):
        if self.latest_depth is None or self.current_odom is None: return

        if self.state == 'IDLE':
            if time.time() - self.idle_start_time > 2.0:
                self.start_pose = self.current_odom.pose.pose.position
                self.start_orientation = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)
                self.state = 'DETECT'
                self.get_logger().info('Ready to detect Kitchen Tools!')

        elif self.state == 'DETECT':
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(cv_image, stream=True, conf=0.5, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    dist = self.latest_depth[cy, cx] / 1000.0
                    
                    if 0.3 < dist < 3.0:
                        if abs(self.prev_dist - dist) < 0.02: self.stop_count += 1
                        else: self.stop_count = 0
                        self.prev_dist = dist

                        if self.stop_count > 5:
                            self.target_angle = -(cx - 320) / 640.0 * 1.2
                            self.target_dist = dist
                            self.trigger_sound()
                            self.state = 'ROTATE_TO_OBJ'

        elif self.state == 'ROTATE_TO_OBJ':
            self.send_control(0.0, 0.4 if self.target_angle > 0 else -0.4)
            time.sleep(abs(self.target_angle) * 1.3) 
            self.send_control(0.0, 0.0)
            self.move_start_pose = self.current_odom.pose.pose.position
            self.state = 'MOVING'

        elif self.state == 'MOVING':
            dx = self.current_odom.pose.pose.position.x - self.move_start_pose.x
            dy = self.current_odom.pose.pose.position.y - self.move_start_pose.y
            if math.sqrt(dx**2 + dy**2) < (self.target_dist - 0.25):
                self.send_control(0.12, 0.0)
            else:
                self.send_control(0.0, 0.0)
                self.wait_start_time = time.time()
                self.state = 'WAIT'

        elif self.state == 'WAIT':
            if time.time() - self.wait_start_time > 2.0:
                self.state = 'RETURN'

        elif self.state == 'RETURN':
            rx = self.start_pose.x - self.current_odom.pose.pose.position.x
            ry = self.start_pose.y - self.current_odom.pose.pose.position.y
            dist_to_home = math.sqrt(rx**2 + ry**2)
            
            if dist_to_home > 0.05:
                self.send_control(-0.12, 0.0)
            else:
                current_yaw = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)
                yaw_diff = self.start_orientation - current_yaw
                if abs(yaw_diff) > 0.1:
                    self.send_control(0.0, 0.4 if yaw_diff > 0 else -0.4)
                else:
                    self.send_control(0.0, 0.0)
                    self.state = 'IDLE'
                    self.idle_start_time = time.time()
                    self.get_logger().info('Returned Home and Re-aligned!')

    def trigger_sound(self):
        s_msg = Sound()
        s_msg.value = Sound.ERROR
        self.sound_pub.publish(s_msg)

    def send_control(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

def main():
    rclpy.init()
    node = KitchenToolsFinalNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()
