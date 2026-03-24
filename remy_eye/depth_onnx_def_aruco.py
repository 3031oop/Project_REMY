# 전처리 및 후처리 추가
# import cv2, torch, pandas, numpy as np, threading, queue, ncnn, sounddevice as sd, scipy.io.wavfile as wav
import cv2, torch, pandas, numpy as np, threading, queue, ncnn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import RunningMode
# import RPi.GPIO as GPIO
import time,math
from collections import deque
import pyrealsense2 as rs
import select
import onnxruntime as ort
from ultralytics import YOLO
import os
import queue
import socket
import threading
import re


send_queue = queue.Queue()


current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(current_dir, 'yolo11n-pose.pt')
yolo_model = YOLO(yolo_model_path)
# --- 상태 관리 변수 ---
is_person_present = False # 사람이 있는지 없는지
pending_zone = -1        # 현재 머물고 있는 새로운 구역
last_sent_zone = -1       # 로봇에게 마지막으로 전송한 구역 
zone_entry_time = 0     # 사람이 카메라에 잡힌 시간
STABILIZE_DELAY = 3   # 구역 안정화 시간 
SCAN_DURATION = 3.0
current_state = 0 # 0: 대기, 1: 도마, 2: 스토브
scan_step = 0
scan_timer = 0
ABSENT_THRESHOLD = 3.0
DIST_THRESHOLD = 0.02
LOST_HAND_THRESHOLD = 150 # 30프레임(약 1초) 동안 손이 안 보이면 자리를 비운 것으로 간주
hand_unseen_counter = 0   # 손이 안 보인 프레임을 카운트하는 변수
current_target_id = None
aruco_sent = False
prev_event = ""
# ========================


# ====== 소켓 ======
#HOST = "127.0.0.1"
HOST = "10.10.141.126" 
PORT = 5000

recvFlag = False
rsplit = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def gettingMsg(): 
    global rsplit, recvFlag, TARGET, target_pixel, current_target_id, aruco_sent
    
    while True: 
        # s(소켓)에 읽을 데이터가 있는지 딱 0.01초만 확인
        try:
            readable, _, _ = select.select([s], [], [], 0.01)
            if readable:
                data = s.recv(1024) 
                if not data: 
                    print(">>> 서버와의 연결이 끊겼습니다.")
                    break
                
                rstr = data.decode("utf-8")
                print(f"DEBUG RECEIVE: {rstr}")
                # 메시지가 있을 때만 정규식 연산 실행 (CPU 절약)
                if rstr.strip():
                    rsplit = re.split('[\]|\[@]|\n', rstr)
                    print(f">>> [THREAD] recvFlag를 True로 변경함! (rsplit: {rsplit})")
                    recvFlag = True

                    if "ID@" in rstr.lower():
                        # 정규식으로 id@ 뒤의 숫자 추출
                        match = re.search(r'ID@(\d+)', rstr.lower())
                        if match:
                            try:
                                current_target_id = int(match.group(1))
                                aruco_sent = False
                                print(f"🎯 ArUco 타겟 ID 고정: {current_target_id}")
                            except: pass
                            

                    elif "stop" in rstr.lower():
                        current_target_id = None
                        aruco_sent = False
                        print(f"ArUco 추적 중지")

                    if "TARGET@" in rstr:
                        match = re.search(r'TARGET@(\d+)', rstr)
                        if match:
                            new_idx = int(match.group(1))
                            TARGET = new_idx
                            target_pixel = [] # 타겟 변경 시 픽셀 초기화
                            print(f">>> PC 타겟 변경 수신: {TOOLS_NAME[TARGET]}")
                            
        except Exception as e:
            print(f"{e}")
            break


# =====onnx==========
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4
session = ort.InferenceSession("kitchen_model_v2_240_424.onnx", providers=['CPUExecutionProvider'])
input_name = "images"
output_name = "output0"

DEBUG_DRAW=True
IMG_W = 448
IMG_H = 256
CONF_TH = 0.6
FRAME_SKIP = 2
IOU_TH = 0.25
DETECT_PADDING = 10

WARNING_TXT_ORG = (30,30)
DERECTION_ORG = (430,450)
TARGET_TXT_ORG=(30,65)
COLOR_RED = (0,0,255)
COLOR_GREEN = (0,255,0)
COLOR_BLUE = (255,0,0)
TARGET=0


LED_NUMBER = [2,3,4,14,15,18,17,27,22]
TOOLS_NAME = ["Knife", "Fork", "Ladle", "Plate"]
CHUNK=512

fid = 0
target_pixel = []
target_pixel_ttl=0
blade_pixel = []
blade_pixel_ttl=0
flag_target=False
flag_blade=False

last_blade_z = 0.0
prev_led_msg = ""


# 전송 전담 스레드 함수
def socket_send_worker():
    while True:
        msg = send_queue.get()
        # 큐에 메시지가 너무 많이 쌓였다면 최신 것만 남기고 버림
        while not send_queue.empty():
            msg = send_queue.get()
            send_queue.task_done()
        try:
            s.send(msg.encode())
        except:
            pass
        send_queue.task_done()


# ===== 그리기 =====
def draw_landmarks_on_image(tools_dect, hand_dect, frame):
    # --- 추가된 안전장치: hand_dect가 None이면 그림을 그리지 않고 원본 반환 ---
    if hand_dect is None or not hasattr(hand_dect, 'hand_landmarks'):
        return np.copy(frame)
    landmarks_list = hand_dect.hand_landmarks
    annotated_image = np.copy(frame)

    # 도구 박스
    for (x1,y1,x2,y2), s, _ in tools_dect:
        cv2.rectangle(annotated_image, (x1,y1), (x2,y2), (0,0,255), 2)
    
    hand_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
    for style in hand_style.values():
        style.circle_radius = 2
        # style.thickness = 1
    
    # 손 랜드마크
    for landmark in landmarks_list:
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in landmark:
            landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)])

        if DEBUG_DRAW:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                hand_style, # 크기만 줄인 알록달록 스타일
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            # solutions.drawing_utils.draw_landmarks(annotated_image,landmarks_proto,solutions.hands.HAND_CONNECTIONS,solutions.drawing_styles.get_default_hand_landmarks_style())
    draw_text(annotated_image,TOOLS_NAME[TARGET], TARGET_TXT_ORG,(255,255,0))
    return annotated_image
def setting_target(tools_result):
    global TOOLS_NAME, TARGET, target_pixel,target_pixel_ttl,fid
    if TOOLS_NAME[TARGET] == "Knife": target = 2
    elif TOOLS_NAME[TARGET] == "Fork": target = 1
    elif TOOLS_NAME[TARGET] == "Ladle": target = 3
    elif TOOLS_NAME[TARGET] == "Plate": target = 4
    else: target = 2
    target_xy = [ tools[0] for tools in tools_result if tools[2] == target ]
    if len(target_xy) > 0:
        target_pixel = save_pixel(target_xy)
        target_pixel_ttl=fid

def inside_allhand(hand_pixel,target_pixel,W,H,depth_frame,threshold_z=0.05):
    # global last_blade_z
    min_dz = 99.9

    if not target_pixel or not hand_pixel:
        return False, 0.0
    
    target = target_pixel[0]
    tx, ty = np.clip((target[0] + target[2]) // 2, 0, W-1), np.clip((target[1]+target[3]) // 2, 0, H-1)
    target_z = depth_frame.get_distance(tx,ty)

    if target_z <= 0: return False,0.0

    for lm in hand_pixel:
        lm_x, lm_y = int(lm.x * W), int(lm.y * H)

        for target in target_pixel:
            if check_inside(target, lm_x,lm_y):
                hand_z = depth_frame.get_distance(lm_x,lm_y)

                if hand_z <= 0: continue
                blade_dist = abs(hand_z - target_z)

                if blade_dist < threshold_z:
                    return True, blade_dist

    # 충돌 안 했을 때: (False, 관측된 최소 거리) 반환
    return False, min_dz

def detection_box(tools_dect, hand_dect, current_frame, depth_frame, intr):
    target_id = "[VI]"
    global prev_blade_pos, prev_led_msg

    blade_xy  = [ tools[0] for tools in tools_dect if tools[2] == 0 ]
    # setting_target(tools_dect)
    setting_target(tools_result=tools_dect)

    global target_pixel, blade_pixel,flag_target,flag_blade,blade_pixel_ttl,fid, prev_event
    if len(blade_xy)  > 0:
        blade_pixel  = save_pixel(blade_xy)
        blade_pixel_ttl=fid
    annotated_image = np.copy(current_frame)

    hand_list = hand_dect.hand_landmarks
    H, W, _ = annotated_image.shape

    flag_Up = False; flag_Down = False; flag_Left = False; flag_Right = False
    
    if not hand_list:
        flag_blade=False
        flag_target=False
    hand_pixel=[]

    # 손잡이 확인
    for hand in hand_list:
        # 손의 중심점(8번 마디) 픽셀 및 3D 좌표
        mx, my = int(np.clip(hand[8].x * W, 0, W-1)), int(np.clip(hand[8].y * H, 0, H-1))
        m_depth = depth_frame.get_distance(mx, my)
        cv2.circle(annotated_image, (mx, my), 2, COLOR_BLUE, -1)
        
        for lm in hand: hand_pixel.append(lm)
        if m_depth <= 0: continue
        h_point = rs.rs2_deproject_pixel_to_point(intr, [mx, my], m_depth)

        for target in target_pixel:
            tx = int(np.clip((target[0] + target[2]) // 2, 0, W-1))
            ty = int(np.clip((target[1] + target[3]) // 2, 0, H-1))
            t_depth = depth_frame.get_distance(tx, ty)
            if t_depth <= 0: continue
            t_point = rs.rs2_deproject_pixel_to_point(intr, [tx, ty], t_depth)

            # --- [3D 좌표 차이 계산] ---
            # diff_x: 좌우(카메라 기준), diff_y: 상하(카메라 기준), diff_z: 깊이(카메라 기준)
            dx = t_point[0] - h_point[0]
            dy = t_point[1] - h_point[1]
            dz = t_point[2] - h_point[2]    

            # 주의: 카메라는 90도 돌아가 있음
            cam_Up = (dz - dy) > DIST_THRESHOLD
            cam_Down = (dz - dy) < -DIST_THRESHOLD
            cam_Left = (dx + dz) < -DIST_THRESHOLD
            cam_Right = (dx + dz) > DIST_THRESHOLD

            # 2. 카메라 기준 방향을 "사람 기준 실제 방향"으로 변환합니다. (90도 회전 예시)
            # 여기서만 수정을 하면, 아래 로직은 건드릴 필요가 없습니다.
            flag_Right = cam_Up     # 카메라의 위쪽이 실제 나의 오른쪽일 때
            flag_Left  = cam_Down
            flag_Up    = cam_Left
            flag_Down  = cam_Right

    flag_target, _= inside_allhand(hand_pixel,target_pixel,W,H,depth_frame,0.05)
    flag_blade, blade_dist = inside_allhand(hand_pixel,blade_pixel,W,H, depth_frame,0.05)

    try:
        curr_vi_msg = ""
        curr_voi_msg = ""

        if flag_blade:
            draw_text(annotated_image, "DANGER! ", WARNING_TXT_ORG, COLOR_RED)
            curr_voi_msg = "[VOI]knife_danger\n"
            curr_vi_msg = f"{target_id}DANGER\n"
            print(f"danger")             
        elif flag_target:
            draw_text(annotated_image, "DETECTED!", WARNING_TXT_ORG, COLOR_GREEN)
            curr_voi_msg = "[VOI]knife_detected\n"
            curr_vi_msg = f"{target_id}DETECTED\n"
            print(f"detected")
        elif hand_list:
            if flag_Left:
                if flag_Up:    curr_vi_msg = f"{target_id}LED@22\n"
                elif flag_Down:curr_vi_msg = f"{target_id}LED@17\n"
                else:          curr_vi_msg = f"{target_id}LED@27\n"
            elif flag_Right:
                if flag_Up:    curr_vi_msg = f"{target_id}LED@4\n"
                elif flag_Down:curr_vi_msg = f"{target_id}LED@2\n"
                else:          curr_vi_msg = f"{target_id}LED@3\n"
            elif flag_Up:      curr_vi_msg = f"{target_id}LED@18\n"
            elif flag_Down:    curr_vi_msg = f"{target_id}LED@14\n"



        # 아무 상태도 아닐 때 상태 초기화 (다시 감지될 수 있도록)
        # if prev_led_msg in [f"{target_id}DANGER\n", f"{target_id}DETECTED\n"]:       
        if curr_vi_msg != prev_led_msg:
            if curr_vi_msg != "":
                # s.send(current_event_msg.encode())
                send_queue.put(curr_vi_msg)
                print(f"LED 전송: {curr_vi_msg.strip()}")
            else: 
                # s.send(f"{target_id}OFF\n".encode())
                send_queue.put(f"{target_id}OFF\n")
                # print("상태 초기화")
            
            prev_led_msg = curr_vi_msg # 현재 상태 저장
        
        if curr_voi_msg != prev_event:
            if curr_voi_msg != "":
                send_queue.put(curr_voi_msg)
                print(f"EVENT 전송: {curr_voi_msg.strip()}")
            prev_event = curr_voi_msg
                


    except Exception as e:
        # 전송 실패 시 조용히 넘어감 (서버 연결 끊김 대비)
        print(f"Error: {e}")

    return annotated_image

def save_pixel(boxes):
    return [[x1,y1,x2,y2] for x1,y1,x2,y2 in boxes]

def check_inside(box, x, y):
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])

def draw_text(image, text, org, color):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

def false_LED():
    global LED_NUMBER
    for led in LED_NUMBER:
        GPIO.output(led, False)

def detected_LED():
    global LED_NUMBER
    for led in LED_NUMBER:
        GPIO.output(led, True)
def button_callback(channel):
    global TARGET, TOOLS_NAME,target_pixel
    if TARGET < len(TOOLS_NAME):
        TARGET += 1
    if TARGET >= len(TOOLS_NAME):
        TARGET = 0
    target_pixel=[]
# ===== 전처리 =====
def letterbox(img, new_shape=(IMG_W, IMG_H), color=(114,114,114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    r = min(new_h / h, new_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    top = (new_h - nh) // 2
    left = (new_w - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def nms(dets, iou_th=IOU_TH):
    if not dets: return []
    boxes  = np.array([d[0] for d in dets], dtype=np.float32)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i,2] - boxes[i,0]) * (boxes[i,3] - boxes[i,1])
        area_r = (boxes[rest,2] - boxes[rest,0]) * (boxes[rest,3] - boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_th]
    return [dets[k] for k in keep]


def tools_inference(frame_rgb):
    H, W, _ = frame_rgb.shape            # (버그 수정) frame -> frame_rgb
    img_lbx, r, lpad, tpad = letterbox(frame_rgb, new_shape=(IMG_W,IMG_H))

    # 2. 전처리: HWC(0,1,2) -> CHW(2,0,1) 변환 및 배치(batch) 차원 추가
    blob = img_lbx.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0

    # 3. ONNX 추론 실행
    # output0의 형태는 [1, 6300, 10]
    preds = session.run([output_name], {input_name: blob})[0]
    A = preds[0]  # [6300, 10]

    # 4. 데이터 분리 (10개의 값: x, y, w, h, obj_conf, cls1, cls2, cls3, cls4, cls5)
    conf_scores = A[:, 4]  # 객체 존재 확률
    class_scores = np.max(A[:, 5:], axis=1)  # 5개 클래스 중 최대값
    scores = conf_scores * class_scores  # 최종 점수
    cls_ids = np.argmax(A[:, 5:], axis=1)  # 클래스 번호

    # 5. 임계값 필터링
    keep = scores >= CONF_TH
    if not np.any(keep): return []

    A = A[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # 6. 좌표 복원 (중심점 -> 박스)
    cx, cy, w, h = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    x1 = (cx - w/2 - lpad) / r
    y1 = (cy - h/2 - tpad) / r
    x2 = (cx + w/2 - lpad) / r
    y2 = (cy + h/2 - tpad) / r

    # 7. 결과 정리 및 NMS 적용
    dets = []
    for i in range(len(scores)):
        rx1, ry1 = max(0, min(x1[i], W)), max(0, min(y1[i], H))
        rx2, ry2 = max(0, min(x2[i], W)), max(0, min(y2[i], H))
        dets.append(([int(rx1), int(ry1), int(rx2), int(ry2)], float(scores[i]), int(cls_ids[i])))

    if dets:
        dets = nms(dets, IOU_TH) # 기존 만드신 nms 함수 호출
    return dets

# =======3/19 사람 인식 함수 =======
def get_valid_person(frame, yolo_model, conf_threshold=0.6):
    # YOLO 추적 실행 (최대 1명)
    results = yolo_model.track(source=frame, persist=True, max_det=1, verbose=False)

    has_person = False
    center_x = -1 # 사람을 못 찾았을 때의 기본값

    # 결과가 있고 박스가 존재하는지 확인
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

         # 확률이 높고(사람인 게 확실하고) 클래스가 0(사람)이며 키포인트가 있을 때
        if conf >= conf_threshold and cls == 0 and results[0].keypoints is not None:
            has_person = True

            # 키포인트 데이터에서 유효한 x값들의 평균 계산
            kp_data = results[0].keypoints.xy[0]
            valid_x = kp_data[kp_data[:,0] > 0][:,0]
            if len(valid_x) > 0:
                center_x = np.mean(valid_x.cpu().numpy() if hasattr(valid_x, 'cpu') else valid_x)
    return has_person, center_x, results


# ===== 메인 =====
def main():
    try:
        s.connect((HOST, PORT))
        # s.send('[REMY:PASSWD]'.encode())
        send_queue.put("[EYE:PASSWD]")
        threading.Thread(target=gettingMsg, daemon=True).start()
        threading.Thread(target=socket_send_worker, daemon=True).start()
        print(">>> 서버 연결 성공!")
        send_queue.put("[OMXA]MOVE@0\n")
    except: print(">>> 서버 연결 실패 (통신 제외 실행)")

     # ===== 3/11 필터 =======
    # 필터 객체 생성
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)

    spatial = rs.spatial_filter()     # 주변 픽셀을 이용한 구멍 메우기 및 스무딩 (주변 픽셀 평균화 효과)
    temporal = rs.temporal_filter()   # 프레임 간의 노이즈를 억제 (Temporal Filter)
    hole_filling = rs.hole_filling_filter() # 뎁스가 비어있는 곳을 채움
    # 필터 옵션 조절 (필요시)
    spatial.set_option(rs.option.holes_fill, 3) # 5x5 주변 픽셀 참조

    threshold_filter = rs.threshold_filter()
    threshold_filter.set_option(rs.option.min_distance, 0.1) # 최소 10cm
    threshold_filter.set_option(rs.option.max_distance, 3.0) # 최대 3m (필요에 따라 조절)
    # ========================

    # 손 랜드마크
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2,running_mode=RunningMode.VIDEO )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    # RealSense 뎁스-컬러 정렬 설정
    
    align_to = rs.stream.color # 뎁스를 컬러에 맞추겠다는 설정
    align = rs.align(align_to)

    # RealSense 설정 시작
    pipeline = rs.pipeline()
    config = rs.config()

    # 해상도 변경
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30) # 뎁스 스트림 추가
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.frames_queue_size):
        depth_sensor.set_option(rs.option.frames_queue_size, 2)
    color_sensor = device.first_color_sensor()
    if color_sensor.supports(rs.option.auto_exposure_priority):
        color_sensor.set_option(rs.option.auto_exposure_priority, 0)
    profile = pipeline.start(config)
    # 픽셀 좌표를 3D로 바꾸기 위한 내적 파라미터(Intrinsics) 가져오기
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # ===== 필터 추가 =====
    # pipeline.start(config) 이후에 추가
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    # 프리셋을 'High Density'로 설정 (먼거리 구멍 메우기에 유리)
    # 0: Custom, 1: High Accuracy, 2: High Density, 3: Medium Density...
    depth_sensor.set_option(rs.option.visual_preset, 2) 

    # 레이저 파워 올리기 (먼거리 반사율 상승)
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, 360) # 최대값 근처로 설정

    global fid,blade_pixel_ttl,blade_pixel,target_pixel,target_pixel_ttl
    global current_state, last_sent_zone, pending_zone, zone_entry_time
    global recvFlag, rsplit, prev_led_msg 
    global scan_step,scan_timer,hand_unseen_counter
    global current_target_id, aruco_sent


   # 최근 30프레임 평균
    t_prev = time.time()

    fid_log=0
    disp_fps = 0.0
    cur_fps = 0.0

    tools_result = [] 
    hand_result = None
    is_stove_on = False
    last_person_seen_time = time.time()
    is_cooking = False

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    try:
        while True:
            if recvFlag:    
                print(f"rsplit: {rsplit}")

                if "START" in rsplit:
                    is_cooking = True
                    print(">> [SYSTEM] 조리시작! 시퀀스를 가동합니다.")
                    send_queue.put("[OMXA]MOVE@0\n")
                elif "FINISH" in rsplit:
                    is_cooking = False
                    print(">> [SYSTEM] 조리 종료. 대기 모드로 전환합니다.")
                    current_state = 0
                    send_queue.put("[OMXA]MOVE@0\n")
                    
                if "STOVE" in rsplit:
                    print(f">>> [MAIN] recvFlag 감지됨! rsplit 데이터: {rsplit}")
                    if "ON" in rsplit:
                        is_stove_on = True
                        print(f">> 스토브 ON,{is_stove_on}")
                        if current_state == 0:
                            current_state = 2
                            send_queue.put(f"[OMXA]MOVE@2\n")
                    elif "OFF" in rsplit:
                        is_stove_on = False
                        print(">> 스토브 OFF")
                        scan_step = 0
                recvFlag = 0


            
            
            # 조리 시작
            t0 = time.time()
            frames = pipeline.poll_for_frames()

            if not frames:
                continue
            
            aligned_frames = align.process(frames)

            temp_color_frame = aligned_frames.get_color_frame()
            temp_depth_frame = aligned_frames.get_depth_frame()
            '''
            temp_color_frame = frames.get_color_frame()
            temp_depth_frame = frames.get_depth_frame()
            '''
            if not temp_color_frame or not temp_depth_frame:
                continue

            filtered_frame = threshold_filter.process(temp_depth_frame)
            filtered_frame = spatial.process(filtered_frame)
            filtered_frame = temporal.process(filtered_frame)
            # filtered_frame = hole_filling.process(filtered_frame)

            depth_frame = filtered_frame.as_depth_frame()
            color_frame = temp_color_frame

            # 리얼센스 데이터를 넘파이 배열로 변환 (기존 코드의 frame 역할)
            frame = np.asanyarray(color_frame.get_data())

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            annotated_image = frame.copy()

            
            # 조리 시작 전
            if not is_cooking:
                cv2.putText(annotated_image, "WAITING FOR 'START' COMMAND...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                annotated_image = cv2.resize(annotated_image, (848 , 480 ))
                cv2.imshow("hand_land", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue # 아래의 무거운 로직(YOLO 등)을 모두 건너뜁니다.

            fid += 1
            # ===== 3/16 사람인식 =====

            if current_state == 0:
                status_text = "SCANNING..."
                text_color = (255, 255, 255)
                if fid % 5 == 0:
                    has_person, center_x, yolo_results = get_valid_person(frame, yolo_model)

                    if has_person:
                        if center_x < 150: target_zone = 1 # 도마
                        elif center_x < 300: target_zone = 0 # 대기
                        else: target_zone = 2              # 스토브

                        # print(f"DEBUG | X: {center_x:.1f} | Zone: {target_zone} | Pending: {pending_zone}")
                        status_text = f"DETECTING - Zone: {target_zone}"
                        color = (0, 255, 0)
                        
                    
                        if target_zone != last_sent_zone:
                            if target_zone != pending_zone:
                                pending_zone = target_zone
                                zone_entry_time = time.time()
                                print(f"--- 구역 변경 감지: {target_zone} (안정화 대기중...) ---")
                            else:
                                elapsed = time.time() - zone_entry_time
                                if elapsed >= STABILIZE_DELAY:
                                    last_sent_zone = target_zone
                                    current_state = target_zone
                                    print(f"★ 확정된 구역: {target_zone} (모드 전환)")
                                    try:
                                        # s.send(f"[OMXA]MOVE@{target_zone}\n".encode())
                                        send_queue.put(f"[OMXA]MOVE@{target_zone}\n")
                                        print(f">>> 로봇 이동 명령 전송 : {target_zone}")
                                    except: pass
                        else:
                            pending_zone = -1

                if 'yolo_results' in locals() and len(yolo_results[0]) > 0:
                    annotated_image = yolo_results[0].plot()
                cv2.putText(annotated_image, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


            # [상태 1: 도마 안전 모드]
            elif current_state == 1:
                status_text = "CUTTING BOARD MODE"

                # 아루코 마커 인식
                # if current_target_id is not None:
                if current_target_id is not None and not aruco_sent:
                    corners, ids, _ = aruco_detector.detectMarkers(frame)
                    if ids is not None:
                        for i in range(len(ids)):
                            if int(ids[i][0]) == current_target_id:
                                u = int(np.mean(corners[i][0][:, 0])) 
                                v = int(np.mean(corners[i][0][:, 1]))
                                z = depth_frame.get_distance(u, v)
                                
                                if z > 0:
                                    point = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                                    #mm 변환
                                    x = point[0] * 1000
                                    y = point[1] * 1000
                                    z = point[2] * 1000
        
                                    aruco_msg = f"[LR]{x:.3f},{y:.3f},{z:.3f}\n"
                                    send_queue.put(aruco_msg)
                                    aruco_sent = True
                                    print(f"📤 ID {current_target_id} 좌표 1회 전송 완료: {x:.3f}, {y:.3f}, {z:.3f}")
                                    print(f"{aruco_sent}")
                                    
                                    # current_target_id = None
                                    cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
                            
                # 손 인식
                if fid % (FRAME_SKIP + 1) == 1:                   
                    ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                    hand_result = hand_detector.detect_for_video(mp_image, ts_ms)
                # 도구 인식(ONNX)
                if fid % 4 == 1:
                    tools_result = tools_inference(frame_rgb)
                
                if blade_pixel and fid-blade_pixel_ttl>5 :
                    blade_pixel=[]
                if target_pixel and fid-target_pixel_ttl>20 :
                    target_pixel=[]

                # 손 감지 여부 판단
                # 손 감지된 경우
                if hand_result is not None and hand_result.hand_landmarks:
                    hand_unseen_counter = 0
                    drawing_image = draw_landmarks_on_image(tools_result, hand_result, frame)
                    annotated_image = detection_box(tools_result, hand_result, drawing_image, depth_frame, intr)
                # 손이 감지되지 않은 경우
                else: 
                    hand_unseen_counter += 1
                    annotated_image = frame.copy()
                    status_text = f"HANDS NOT SEEN ({hand_unseen_counter}/{LOST_HAND_THRESHOLD})"

                    if hand_unseen_counter >= LOST_HAND_THRESHOLD:
                        hand_unseen_counter = 0 # 초기화
                        if is_stove_on:
                            print(">> 도마 모드 중단: 스토브 ON -> 스토브 감시로 전환")
                            current_state = 2
                            send_queue.put(f"[OMXA]MOVE@2\n")
                        else:
                            print(">> 도마 모드 중단: 손 미감지로 인한 기본 위치 복구")
                            send_queue.put(f"[OMXA]MOVE@0\n")
                            current_state = 0
                        
                        last_sent_zone = current_state
                        continue
            
            
            # 2: 스토브 모드
            elif current_state == 2:         
                # 사람 감지 (YOLO 실행)
                if fid % 5 == 0:
                    has_person, _, yolo_results = get_valid_person(frame, yolo_model)
                    print(f"\n[DEBUG STOVE] is_stove_on: {is_stove_on} | has_person: {has_person} | scan_step: {scan_step}")
                    
                    # 사람 감지 시 
                    if has_person:
                        # 마지막 사람 포착 시각
                        last_person_seen_time = time.time()
                        status_text = "STOVE MONITORING ..."
                        text_color = COLOR_GREEN
                        
                        if scan_step > 0:
                            print(f">> [발견] {last_sent_zone}번 구역에서 사용자를 찾았습니다!")
                            current_state = last_sent_zone
                            scan_step = 0
                            scan_timer = 0
                            if current_state == 0:
                                print(">> 기본 모드 시작")
                            continue
                    
                    # 사람이 감지되지 않을 시
                    else:
                        # 사람이 안보인지 얼마나 지났나 계산
                        absent_duration = time.time() - last_person_seen_time

                        if fid % 15 == 0:
                            print(f"\n[DEBUG STOVE] is_stove_on: {is_stove_on} | has_person: {has_person} | scan_step: {scan_step}")
                        if absent_duration > ABSENT_THRESHOLD:
                            if is_stove_on:
                                wait_time = time.time() - scan_timer
                                # 스토브 켜져 있다면 순차 탐색 시작
                                if scan_step == 0:
                                    print(">>> [사람 부재] 대기 구역(0) 확인 시작")
                                    send_queue.put(f"[VOI]stove_user_left\n")
                                    send_queue.put(f"[OMXA]MOVE@0\n")
                                    last_sent_zone = 0
                                    scan_step = 1
                                    scan_timer = time.time()

                                elif scan_step == 1: # 대기(0) 확인 중
                                    
                                    if fid % 15 == 0:
                                        print(f"   ㄴ [SCAN 0] 0번 구역 확인 중... {wait_time:.1f}초 경과")

                                    if wait_time > SCAN_DURATION:
                                        print(">> 0번 부재: 도마로 이동")
                                        send_queue.put(f"[OMXA]MOVE@1\n")
                                        last_sent_zone = 1
                                        scan_step = 2
                                        scan_timer = time.time()

                                elif scan_step == 2: # 도마(1) 확인중
                                    if wait_time > SCAN_DURATION:
                                        print(">> 모든 구역에 사람 없음 스토브 복귀 및 경고")
                                        send_queue.put(f"[OMXA]MOVE@2\n")
                                        # 스토브 경고 보내기
                                        # send_queue.put(f"{target_id}DANGER_STOVE\n")
                                        last_sent_zone = 2
                                        scan_step = 3
                                        # scan_timer = time.time()

                                elif scan_step == 3:
                                    status_text = "!!! WARNING: STOVE UNATTENDED !!!"
                                    text_color = COLOR_RED

                                    final_wait_time  = time.time() - scan_timer

                                    if final_wait_time > 20:
                                        print(">>> 20초 경과 : 스토브 자동 차단 명령 전송")
                                        send_queue.put(f"[VOI]stove_warning\n")
                                        send_queue.put(f"[STOVE]FIRE@OFF\n")
                                        is_stove_on = False
                                        scan_step = 4

                
                            else:
                                if fid % 15 == 0:
                                    print(f"안움직임")
                                # 스토브도 꺼져있고 사람도 없으면 기본 복귀
                                scan_step = 0
                                current_state = 0
                                send_queue.put(f"[OMXA]MOVE@0\n")
                                last_sent_zone = 0

                
                # 드로잉 및 상태 텍스트 표시
                if 'yolo_results' in locals() and len(yolo_results[0]) > 0:
                    annotated_image = yolo_results[0].plot()
                cv2.putText(annotated_image, status_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


                
            # FPS 그리기
            t_now = time.time()
            elapsed_time = t_now - t_prev
            if elapsed_time > 1.0: # 1초가 지날 때마다
                cur_fps = (fid - fid_log) / elapsed_time # (현재 프레임 - 1초 전 프레임) / 시간
                fid_log = fid
                t_prev = t_now
            h, w, _ = annotated_image.shape
            cv2.putText(annotated_image, f"FPS: {cur_fps:.1f}", (w - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            annotated_image = cv2.resize(annotated_image, (848 , 480 ))
            cv2.imshow("hand_land", annotated_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

