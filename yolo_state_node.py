import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from turtlebot3_msgs.msg import Sound
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
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 6)
        profile =self.pipeline.start(config)

        #ready time
        time.sleep(1.0)


        #lazer power & freeset setup
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 2) # High Density
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, 360)

        #remove noise 
        #self.threshold_filter = rs.threshold_filter(0.1, 4.0)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.align = rs.align(rs.stream.color) 


        # 3) State & Detection Variables
        self.state = 'IDLE'
        self.prev_sent_state = None
        self.current_odom = None
        self.latest_target = None  # Shared detection result
        self.lock = threading.Lock()

        # Home & Path Variables
        self.start_pose = None
        self.start_orientation = 0.0
        self.move_start_pose = None
        self.stop_count = 0
        self.prev_dist = 0.0
        self.target_dist = 0.0
        self.sound_count = 0
        self.last_error = 0

        #Patrol variables
        self.patrol_sub_state = 'MOVE'
        self.patrol_start_pose = None
        self.patrol_start_yaw = 0.0
        self.rotate_end_time = 0.0
        self.found_during_patrol = False
        self.wait_start_time =0.0;       


        # 4) ROS Pub/Sub
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        #self.sound_pub = self.create_publisher(Sound, '/sound', 10)
        self.image_out_pub = self.create_publisher(Image, '/image_out', 10)
        self.command_pub = self.create_publisher(String, '/waffle_command', 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.command_sub = self.create_subscription(String, '/waffle_command', self.command_callback, 10)
        self.sound_client = self.create_client(SoundSrv, '/sound')


        # 5) Start Threads & Timers
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
        self.timer = self.create_timer(0.1, self.main_control_loop) # 10Hz Control


    #callback function
    def odom_callback(self, msg):
        self.current_odom = msg

    def command_callback(self, msg):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received command: {command}")

        #filltering
        #if command in ['patrol', 'detect', 'depart', 'wait_return', 'force_return', 'return']:
        if command in ['patrol', 'detect', 'depart', 'wait_return', 'force_return']:
            return


        if command == 'start':
            if self.state == 'IDLE':
                self.get_logger().info("Remote START(PATROL): Beginning detection...")
                self.state = 'PATROL'
                self.patrol_sub_state = 'MOVE'
                self.patrol_start_pose = None
                self.rotate_end_time = 0.0
                self.stop_count = 0
            else:
                self.get_logger().warn(f"Ignored 'start': Robot is already in {self.state}")

        #elif command == 'return':
        if command == 'return':
                self.get_logger().info("Remote RETURN: Stopping current task and going home.")
                self.send_control(0.0, 0.0)
                
                #self.wait_start_time = time.time() + 999999.0
                self.state = 'RETURN'
                self.wait_start_time = None
                self.prev_sent_state = 'return'
                #self.state = 'RETURN'

        #elif command == 'force_return':
        #    self.get_logger().info("Remote FORCE_RETURN: Emergency return initiated!")
        #    self.send_control(0.0, 0.0)
        #    self.sound_count = 0
        #    self.state = 'RETURN'

    #logic function
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_current_xy(self):
        p = self.current_odom.pose.pose.position
        return float(p.x), float(p.y)

    # --- Background Inference Thread ---
    def inference_loop(self):
        """Background thread for YOLO inference to keep high FPS."""
        time.sleep(3.0)

        while rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if not frames: continue

                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame: continue

                #color_image = np.asanyarray(color_frame.get_data())
                
                #filtered_depth = self.threshold_filter.process(depth_frame) #(0.1~4.0m delete)
                filtered_depth = self.spatial.process(depth_frame)
                #filtered_depth = self.temporal.process(filtered_depth)# frame smooth
                filtered_depth = self.hole_filling.process(filtered_depth)
                
                refined_depth = filtered_depth.as_depth_frame()
                color_image = np.asanyarray(color_frame.get_data())

                #image save
                #current_det = {'cx': None, 'dist': 0.0, 'img': color_image.copy()}

                # YOLO Predict
                results = self.model.predict(source=color_image, conf=0.3, imgsz=320, device='cpu',half=False, verbose=False)
                
                best_det = None
                if results and len(results[0].boxes) > 0:
                    r = results[0]
                    # Get the highest confidence box
                    box = r.boxes[0]
                    xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                    cx, cy = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)

                    if 0<= cx < self.width and 0 <= cy < self.height:
                        dist = float(refined_depth.get_distance(cx, cy))

                    if 0.3 < dist < 3.0:
                        debug_img = color_image.copy()
                        cv2.rectangle(debug_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        best_det = {'cx': cx, 'dist': dist, 'img': debug_img}

                if best_det is None:
                        best_det = {'cx': None, 'dist': 0.0, 'img': color_image.copy()}

                with self.lock:
                    self.latest_target = best_det

            except Exception as e:
                self.get_logger().warn(f"Inference Thread Error: {e}")

            time.sleep(0.05)

    # --- Main Control Loop (10Hz) ---
    def main_control_loop(self):
        if self.current_odom is None: return
        
        # Get latest detection safely
        with self.lock:
            target = self.latest_target

        #display_image = None
        if target is None: return
        
        display_image = target['img']

        # --- State Machine ---
        if self.state == 'IDLE':
            self.start_pose = self.get_current_xy()
            self.start_orientation = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)
            #self.state = 'PATROL'

        elif self.state == 'PATROL':
            #PATROL msg publish
            #status_msg = String()
            #status_msg.data = "patrol"
            #self.command_pub.publish(status_msg)
            #self.get_logger().info("Published: patrol")
            self.publish_msgs("patrol");


            if target['cx'] is not None:
                error = (self.width / 2) - target['cx']
                
                if abs(error) > 50:
                    slow_down_speed = max(0.12, abs(error) * 0.001) 
                    self.send_control(0.0, slow_down_speed if error > 0 else -slow_down_speed)
                    self.get_logger().info("Target spotted! Slowing down and centering...")
                else:
                    self.send_control(0.0, 0.0)
                    self.state = 'DETECT'
                return

            cur_x, cur_y = self.get_current_xy()

            if self.patrol_sub_state == 'MOVE':
                if self.patrol_start_pose is None:
                    self.patrol_start_pose = (cur_x, cur_y)
                
                dist = math.sqrt((cur_x - self.patrol_start_pose[0])**2 + (cur_y - self.patrol_start_pose[1])**2)
                if dist < 1.0:
                    self.send_control(0.07, 0.0)
                else:
                    self.send_control(0.0, 0.0)
                    self.patrol_sub_state = 'ROTATE'
                    #self.rotate_end_time = time.time() + 12.6 # 0.5rad/s -> 1cycle time 12.6
                    self.rotate_end_time = time.time() + 21.0 # 0.5rad/s -> 1cycle time 12.6

            elif self.patrol_sub_state == 'ROTATE':
                remaining = self.rotate_end_time - time.time()
                if remaining > 0:
                    self.send_control(0.0, 0.3)
                else:
                    self.send_control(0.0, 0.0)
                    self.get_logger().warn("Patrol finished without detection. Returning home.")
                    self.state = 'RETURN'

        elif self.state == 'DETECT':
            if target['cx'] is not None:
                if abs(self.prev_dist - target['dist']) < 0.03: 
                    self.stop_count += 1
                else:
                    self.stop_count = 0
                
                self.prev_dist = target['dist']

                if self.stop_count > 5:

                    #detect msg publish
                    #status_msg = String()
                    #status_msg.data = "detect"
                    #self.command_pub.publish(status_msg)
                    #self.get_logger().info("Published: detect")
                    self.publish_msgs("detect");

                    #rotate ready logic
                    #85.2 -> 1.487rad
                    HFOV_RAD = 1.487
                    self.target_dist = target['dist']
                    target_angle = -((target['cx'] - (self.width / 2)) / float(self.width)) * HFOV_RAD
                    self.rotate_cmd = 0.4 if target_angle > 0 else -0.4
                    self.rotate_end_time = time.time() + abs(target_angle) * 1.3
                    #self.trigger_sound()
                    self.state = 'ROTATE'
                    self.stop_count = 0

        #elif self.state == 'ROTATE':
        #    if time.time() < self.rotate_end_time:
        #        self.send_control(0.0, self.rotate_cmd)
        #    else:
        #        self.send_control(0.0, 0.0)
        #        self.move_start_pose = self.get_current_xy()
        #        self.sound_count = 50
        #        self.state = 'MOVING'

        elif self.state == 'ROTATE':
            if target['cx'] is not None:
                error = (self.width / 2) - target['cx']
                self.last_error = error

                if abs(error) > 30:
                    angular_z = error * 0.0008

                    min_speed = 0.12
                    if abs(angular_z) < min_speed:
                        angular_z = min_speed if error > 0 else -min_speed

                    angular_z = max(-0.2, min(0.2, angular_z))
                    self.send_control(0.0, angular_z)
                else:
                    self.send_control(0.0, 0.0)
                    self.get_logger().info("Target Aligned")
                    self.move_start_pose = self.get_current_xy()
                    self.sound_count = 50
                    self.state = 'MOVING'
            else:
                if abs(self.last_error) > 20:
                    blind_speed = 0.12 if self.last_error > 0 else -0.12
                    self.send_control(0.0, blind_speed)
                else:
                    self.send_control(0.0, 0.0)
                    self.state = 'DETECT'

        elif self.state == 'MOVING':
            cur_x, cur_y = self.get_current_xy()
            moved = math.sqrt((cur_x - self.move_start_pose[0])**2 + (cur_y - self.move_start_pose[1])**2)
            
            if moved < max(0.0, self.target_dist - 0.3):
                self.send_control(0.1, 0.0)
            
            #5s, 0.2hz sound buzzer
                self.sound_count += 1
                if self.sound_count >= 30:
                    if self.sound_client.service_is_ready():
                        req = SoundSrv.Request()
                        req.value = 3
                        self.sound_client.call_async(req)
                    self.sound_count = 0
            else:
                self.send_control(0.0, 0.0)
                self.sound_count = 0
                self.wait_start_time = time.time()
                
                #depart msg publish
                #status_msg = String()
                #status_msg.data = "depart"
                #self.command_pub.publish(status_msg)
                #self.get_logger().info("Published: depart")
                self.publish_msgs("depart");
                self.state = 'WAIT'

        elif self.state == 'WAIT':
            #if self.state != 'WAIT':
            #    self.state = 'RETURN'
            #    return

            if self.wait_start_time is not None:
                self.publish_msgs("wait_return");

                elapsed = time.time() - self.wait_start_time

                #if time.time() - self.wait_start_time > 10.0:
                if elapsed  > 10.0:
                    #force_return
                    self.get_logger().warn("Timeout! Force returning...")
                    self.publish_msgs("force_return");
                    self.state = 'RETURN'
                    self.start_time = None
            else:
                pass
        #elif self.state == 'RETURN':
        #    cur_x, cur_y = self.get_current_xy()
        #    remain = math.sqrt((self.start_pose[0] - cur_x)**2 + (self.start_pose[1] - cur_y)**2)
        #    if remain > 0.05:
        #        self.send_control(-0.1, 0.0)
        #    else:
        #        self.state = 'ALIGN_FINAL'

        elif self.state == 'RETURN':

            #status_msg = String()
            #status_msg.data = "return"
            #self.command_pub.publish(status_msg)
            #self.get_logger().info("Published: return")
            self.publish_msgs("return");

            cur_x, cur_y = self.get_current_xy()
            cur_yaw = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)

            dx = self.start_pose[0] - cur_x
            dy = self.start_pose[1] - cur_y
            dist_to_home = math.sqrt(dx**2 + dy**2)
            angle_to_home = math.atan2(dy, dx)

            angle_diff = (angle_to_home - cur_yaw + math.pi) % (2 * math.pi) - math.pi

            if dist_to_home > 0.05:
                if abs(angle_diff) > 0.2:
                    self.get_logger().info("Returning: Aligning to home direction...")
                    self.send_control(0.0, 0.3 if angle_diff > 0 else -0.4)
                else:
                    self.get_logger().info(f"Returning: Moving forward... {dist_to_home:.2f}m left")
                    self.send_control(0.15, 0.0)
            else:
                self.send_control(0.0, 0.0)
                self.state = 'ALIGN_FINAL'


        elif self.state == 'ALIGN_FINAL':
            cur_yaw = self.get_yaw_from_quaternion(self.current_odom.pose.pose.orientation)
            diff = (self.start_orientation - cur_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > 0.05:
                self.send_control(0.0, 0.3 if diff > 0 else -0.3)
            else:
                self.send_control(0.0, 0.0)
                self.get_logger().info(f"ALIGN_FINAL: GO TO IDLE")
                self.state = 'IDLE'
                self.prev_sent_state = None

        # --- UI Overlay ---
        cv2.putText(display_image, f"STATE: {self.state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if target:
            cv2.putText(display_image, f"DIST: {target['dist']:.2f}m", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        self.image_out_pub.publish(self.bridge.cv2_to_imgmsg(display_image, encoding='bgr8'))

    #def trigger_sound(self):
        #s_msg = Sound(); s_msg.value = Sound.ERROR
        #self.sound_pub.publish(s_msg)

    def send_control(self, linear, angular):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

    def publish_msgs(self, status):
        if self.prev_sent_state != status:
            msg = String()
            msg.data = status
            self.command_pub.publish(msg)
            self.get_logger().info(f"Published Msgs: {status}")
            self.prev_sent_state = status

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
