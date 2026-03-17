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

# =====3/16 사람인식(큐 이용)=====
import queue
send_queue = queue.Queue()
# =============================

# =====3/15 사람인식 =====
from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(current_dir, 'yolo11n-pose.pt')
yolo_model = YOLO(yolo_model_path)
# --- 상태 관리 변수 ---
is_person_present = False # 사람이 있는지 없는지
pending_zone = -1        # 현재 머물고 있는 새로운 구역
last_sent_zone = -1       # 로봇에게 마지막으로 전송한 구역 
zone_entry_time = 0     # 사람이 카메라에 잡힌 시간
STABILIZE_DELAY = 3   # 안정화 시간 (0.5초 동안 머물러야 전송)
current_state = 0 # 0: 대기, 1: 도마, 2: 스토브
# ========================


# =====3/11 소켓 추가 ======
import socket
import threading
import re

HOST = "127.0.0.1"
# HOST = "10.10.141.43" 
PORT = 5000
# TARGET_ID = "[OMXA]"

recvFlag = False
rsplit = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def gettingMsg(): 
    global rsplit, recvFlag, TARGET, target_pixel
    s.setblocking(True) # 데이터가 올 때까지 스레드를 잠재웁니다 (CPU 점유율 0%)
    while True: 
        try:
            data = s.recv(1024) 
            if not data: break
            rstr = data.decode("utf-8")

            rsplit = re.split('[\]|\[@]|\n',rstr)  #'[',']','@' 분리
            recvFlag = True

            if "TARGET@" in rstr:
                match = re.search(r'TARGET@(\d+)',rstr)
                if match:
                    new_idx = int(match.group(1))
                    TARGET = new_idx
                    target_pixel = []
                    print(f">>> PC 타켓 변경 수신: {TOOLS_NAME[TARGET]}")

        except: break



# ===========================

# ===== ncnn =====
net = ncnn.Net()
net.opt.num_threads = 4
net.opt.use_fp16_storage = True
net.opt.use_fp16_arithmetic = True
net.load_param("kitchen_model_v2_320.ncnn.param")
net.load_model("kitchen_model_v2_320.ncnn.bin")
DEBUG_DRAW=True
IMG_SIZE = 320
CONF_TH = 0.6
FRAME_SKIP = 2
IOU_TH = 0.25
DETECT_PADDING = 30

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

# ===== 3/12 좌표,led ======
# prev_blade_pos = [0, 0, 0] # 이전 좌표 저장
# MOVE_THRESHOLD = 5.0  # 5mm 이상 움직여야 전송

prev_led_msg = ""
# =====================

# ===== 3/16 사람인식(큐 이용) =====
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
# ===================================


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

    # 손 랜드마크
    for landmark in landmarks_list:
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in landmark:
            landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)])
        if DEBUG_DRAW:
            solutions.drawing_utils.draw_landmarks(annotated_image,landmarks_proto,solutions.hands.HAND_CONNECTIONS,solutions.drawing_styles.get_default_hand_landmarks_style())
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
# def inside_allhand(hand_pixel,target_pixel,W,H):
#     for lm in hand_pixel:
#         lm_x = int(lm.x * W); lm_y = int(lm.y * H)
#         for target in target_pixel:
#             if check_inside(target, lm_x, lm_y) :
#                 return True
#     return False

# ===== 3/9 inside_allhand 함수 수정 =====
def inside_allhand(hand_pixel,target_pixel,W,H,depth_frame,threshold_z=0.05):
    # global last_blade_z
    min_dz = 99.9

    if not target_pixel or not hand_pixel:
        return False, 0.0
    
    target = target_pixel[0]
    tx, ty = np.clip((target[0] + target[2]) // 2, 0, W-1), np.clip((target[1]+target[3]) // 2, 0, H-1)
    target_z = depth_frame.get_distance(tx,ty)

    if target_z <= 0: return False,0.0


    # if is_blade and len(target_pixel) > 0:
    #     target = target_pixel[0]
    #     tx, ty = (target[0]+ target[2]) // 2, (target[1]+ target[3]) // 2
   
    #     current_z = depth_frame.get_distance(tx,ty)
    #     if current_z > 0:
    #         last_blade_z = current_z

    for lm in hand_pixel:
        lm_x, lm_y = int(lm.x * W), int(lm.y * H)

        for target in target_pixel:
            if check_inside(target, lm_x,lm_y):
                hand_z = depth_frame.get_distance(lm_x,lm_y)

                if hand_z <= 0: continue
                dz = abs(hand_z - target_z)

                if dz < threshold_z:
                    return True, dz
                # if is_blade:
                #     dist_hand = depth_frame.get_distance(lm_x,lm_y)
                #     if dist_hand > 0 and last_blade_z > 0:
                #         dz = abs(dist_hand - last_blade_z)
                #         if dz < min_dz: min_dz = dz # 가장 가까운 마디의 거리 저장
                #         if dz < 0.02:
                #             return True, dz # 충돌 시 현재 거리 반환
                # else:
                #     return True, 0.0
    # 충돌 안 했을 때: (False, 관측된 최소 거리) 반환
    return False, min_dz

# ===== 겹칩 확인 + 오디오 이벤트 리턴 =====

# ===== 좌표 추가 ======
# 1. 3D 좌표 추출 함수 추가
# def get_3d_coordinates(pixel_x, pixel_y, depth_frame, intrinsics):
#     # 해당 픽셀의 거리값 (보통 mm 단위 혹은 m 단위)
#     distance = depth_frame.get_distance(pixel_x, pixel_y)
#     if distance > 0:
#         # 카메라 파라미터를 이용해 픽셀(x,y,dist)을 실제 3D 좌표(x,y,z)로 변환
#         result_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], distance)
#         return result_3d # [X, Y, Z]
#     return None



def detection_box(tools_dect, hand_dect, current_frame, depth_frame, intr):
    target_id = "[VI]"
    global prev_blade_pos, prev_led_msg

    # ====== 3/12 좌표,led =======
    # global prev_blade_pos, prev_led_msg
    # pos_id = "[OMXB]"
    # =======================


    blade_xy  = [ tools[0] for tools in tools_dect if tools[2] == 0 ]
    # setting_target(tools_dect)
    setting_target(tools_result=tools_dect)

    global target_pixel, blade_pixel,flag_target,flag_blade,blade_pixel_ttl,fid
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
    for hand in hand_list:
        middle_x = (int(hand[8].x*W))
        middle_y = (int(hand[8].y*H))
        cv2.circle(annotated_image, (middle_x, middle_y), 5, COLOR_BLUE, -1, cv2.LINE_AA)
        for lm in hand:
            hand_pixel.append(lm)
        # 손잡이 확인
        for target in target_pixel:
            target_middle_x = (target[0] + target[2]) // 2
            target_middle_y = (target[1] + target[3]) // 2
            cv2.circle(annotated_image, (target_middle_x, target_middle_y), 5, (255,255,0), -1, cv2.LINE_AA)

            
            if middle_x < target_middle_x-DETECT_PADDING: flag_Right = True
            elif middle_x > target_middle_x+DETECT_PADDING: flag_Left = True
            if middle_y < target_middle_y-DETECT_PADDING:  flag_Down  = True
            elif middle_y > target_middle_y+DETECT_PADDING: flag_Up    = True

        # 날 확인

    flag_target, _= inside_allhand(hand_pixel,target_pixel,W,H,depth_frame,0.05)
    flag_blade, dz = inside_allhand(hand_pixel,blade_pixel,W,H, depth_frame,0.05)

    # if flag_blade:
    #     draw_text(annotated_image, "DANGER", WARNING_TXT_ORG, COLOR_RED)
    # elif flag_target:
    #     draw_text(annotated_image, "DETECTED", WARNING_TXT_ORG, COLOR_GREEN)

    # ===== 3/9 수정 =====
    # if flag_blade:
    #     draw_text(annotated_image, "DANGER! ", WARNING_TXT_ORG, COLOR_RED)
    #     s.send(f"{target_id}DANGER\n".encode()) # 위험 신호 전송
    #     print(f"danger")
    # elif flag_target:
    #     draw_text(annotated_image, "DETECTED!", WARNING_TXT_ORG, COLOR_GREEN)
    #     s.send(f"{target_id}DETECTED\n".encode()) # 감지 신호 전송
    #     print(f"detected")

    # if dz <10.0:
    #     debug_text = f"Z-dist: {dz:.3f}m"
    #     cv2.putText(annotated_image, debug_text, (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    #     # print(f"[DEBUG] 현재 손-칼날 거리: {dz:.3f}m")

    # ===== 3/11 수정 =====
    try:
        current_event_msg = ""
        if flag_blade:
            draw_text(annotated_image, "DANGER! ", WARNING_TXT_ORG, COLOR_RED)
            current_event_msg = f"{target_id}DANGER\n" # 위험 신호 제어
            print(f"danger")             
        elif flag_target:
            draw_text(annotated_image, "DETECTED!", WARNING_TXT_ORG, COLOR_GREEN)
            current_event_msg = f"{target_id}DETECTED\n" # 감지 신호 전송
            print(f"detected")
        elif hand_list:
            if flag_Left:
                if flag_Up:    current_event_msg = f"{target_id}LED@22\n"
                elif flag_Down:current_event_msg = f"{target_id}LED@17\n"
                else:          current_event_msg = f"{target_id}LED@27\n"
            elif flag_Right:
                if flag_Up:    current_event_msg = f"{target_id}LED@4\n"
                elif flag_Down:current_event_msg = f"{target_id}LED@2\n"
                else:          current_event_msg = f"{target_id}LED@3\n"
            elif flag_Up:      current_event_msg = f"{target_id}LED@18\n"
            elif flag_Down:    current_event_msg = f"{target_id}LED@14\n"



            # 아무 상태도 아닐 때 상태 초기화 (다시 감지될 수 있도록)
        # if prev_led_msg in [f"{target_id}DANGER\n", f"{target_id}DETECTED\n"]:       
        if current_event_msg != prev_led_msg:
            if current_event_msg != "":
                # s.send(current_event_msg.encode())
                send_queue.put(current_event_msg)
                print(f"LED 전송: {current_event_msg.strip()}")
            else: 
                # s.send(f"{target_id}OFF\n".encode())
                send_queue.put(f"{target_id}OFF\n")
                # print("상태 초기화")
            
            prev_led_msg = current_event_msg # 현재 상태 저장

        if dz <10.0:
            debug_text = f"Z-dist: {dz:.3f}m"
            cv2.putText(annotated_image, debug_text, (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # print(f"[DEBUG] 현재 손-칼날 거리: {dz:.3f}m")



        
        # print(f"L:{flag_Left} R:{flag_Right} U:{flag_Up} D:{flag_Down}")
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
def letterbox(img, new=IMG_SIZE, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top = (new - nh) // 2
    left = (new - nw) // 2
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

# def tools_inference(frame_rgb):
#     H, W, _ = frame_rgb.shape            # (버그 수정) frame -> frame_rgb
#     img_lbx, r, lpad, tpad = letterbox(frame_rgb, IMG_SIZE)
#     # (버그 수정) img_rgb → img_lbx는 이미 RGB
#     input_mat = ncnn.Mat.from_pixels(img_lbx, ncnn.Mat.PixelType.PIXEL_RGB, IMG_SIZE, IMG_SIZE)
#     input_mat.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])

#     ex = net.create_extractor()
#     ex.input("in0", input_mat)
#     _, result = ex.extract("out0")

#     arr = result.numpy()
#     D, N = arr.shape
#     A = arr.T
    
#     cx, cy, w, h = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
#     cls_scores = A[:, 4:]
    
#     cls_ids = np.argmax(cls_scores, axis=1)
#     scores  = cls_scores[np.arange(A.shape[0]), cls_ids]
    
#     keep = scores >= CONF_TH
#     if not np.any(keep): return []
#     cx = cx[keep]; cy = cy[keep]; w = w[keep]; h = h[keep]
#     scores = scores[keep]; cls_ids = cls_ids[keep]

#     x1 = cx - w/2; y1 = cy - h/2
#     x2 = cx + w/2; y2 = cy + h/2
#     x1 = (x1 - lpad) / r; y1 = (y1 - tpad) / r
#     x2 = (x2 - lpad) / r; y2 = (y2 - tpad) / r

#     x1 = np.clip(x1, 0, W); y1 = np.clip(y1, 0, H)
#     x2 = np.clip(x2, 0, W); y2 = np.clip(y2, 0, H)

#     dets = [([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
#              float(scores[i]), int(cls_ids[i])) for i in range(len(scores))]
#     dets = nms(dets, IOU_TH)
#     return dets

def tools_inference(frame_rgb):
    H, W, _ = frame_rgb.shape
    img_lbx, r, lpad, tpad = letterbox(frame_rgb, IMG_SIZE)
    
    # 1. 입력 설정
    input_mat = ncnn.Mat.from_pixels(img_lbx, ncnn.Mat.PixelType.PIXEL_RGB, IMG_SIZE, IMG_SIZE)
    input_mat.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])

    ex = net.create_extractor()
    ex.input("in0", input_mat)
    ret, result = ex.extract("out0")

    # 2. 데이터를 넘파이로 변환 및 형태 확인
    A = np.array(result) 
    
    # 만약 데이터가 [85, 6300] 처럼 되어있다면 [6300, 85]로 뒤집음
    if A.shape[0] < A.shape[1]:
        A = A.T

    # 3. 확률 계산 (YOLOv5 표준: 객체확률 * 클래스확률)
    # A[:, 4]는 객체가 있을 확률(Objectness)
    # A[:, 5:]는 각 클래스별 확률
    conf_scores = A[:, 4]
    class_scores = np.max(A[:, 5:], axis=1)
    scores = conf_scores * class_scores
    cls_ids = np.argmax(A[:, 5:], axis=1)

    # 4. 임계값(CONF_TH) 필터링
    keep = scores >= CONF_TH
    if not np.any(keep):
        return []

    A = A[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    # 5. 좌표 복원 (중심점 cx, cy, w, h -> x1, y1, x2, y2)
    # imgsz(320) 기준으로 나오므로 letterbox 패딩과 비율을 역계산합니다.
    cx, cy, w, h = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    
    x1 = (cx - w/2 - lpad) / r
    y1 = (cy - h/2 - tpad) / r
    x2 = (cx + w/2 - lpad) / r
    y2 = (cy + h/2 - tpad) / r

    # 6. 결과 리스트 생성
    dets = []
    for i in range(len(scores)):
        # 이미지 범위를 벗어나지 않게 클리핑
        rx1 = max(0, min(x1[i], W))
        ry1 = max(0, min(y1[i], H))
        rx2 = max(0, min(x2[i], W))
        ry2 = max(0, min(y2[i], H))
        
        dets.append(([int(rx1), int(ry1), int(rx2), int(ry2)], float(scores[i]), int(cls_ids[i])))

    # 7. NMS (필수: 안 하면 박스 수백 개 뜹니다)
    if dets:
        dets = nms(dets, IOU_TH)
        
    return dets


# ===== 메인 =====
def main():
    try:
        s.connect((HOST, PORT))
        # s.send('[REMY:PASSWD]'.encode())
        send_queue.put("[REMY:PASSWD]")
        threading.Thread(target=gettingMsg, daemon=True).start()
        threading.Thread(target=socket_send_worker, daemon=True).start()
        print(">>> 서버 연결 성공!")
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

    #=====3/9추가=====
    # 1. RealSense 뎁스-컬러 정렬 설정
    align_to = rs.stream.color # 뎁스를 컬러에 맞추겠다는 설정
    align = rs.align(align_to)
    #================

    # 캠
    # cap = cv2.VideoCapture(0)

    # ======= 추가 =========
    # RealSense 설정 시작
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 뎁스 스트림 추가

    # 해상도 변경
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30) # 뎁스 스트림 추가

    profile = pipeline.start(config)
    # 픽셀 좌표를 3D로 바꾸기 위한 내적 파라미터(Intrinsics) 가져오기
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # ======= 추가 =========

    # ===== 3/11 필터 =====
    # pipeline.start(config) 이후에 추가
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    # 1. 프리셋을 'High Density'로 설정 (먼거리 구멍 메우기에 유리)
    # 0: Custom, 1: High Accuracy, 2: High Density, 3: Medium Density...
    depth_sensor.set_option(rs.option.visual_preset, 2) 

    # 2. 레이저 파워 올리기 (먼거리 반사율 상승)
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, 360) # 최대값 근처로 설정

    global fid,blade_pixel_ttl,blade_pixel,target_pixel,target_pixel_ttl
    global current_state, last_sent_zone, pending_zone, zone_entry_time
    global recvFlag, rsplit, prev_led_msg 

    # if not cap.isOpened():
    #     print("웹 캠을 열 수 없습니다")

    # GPIO.setmode(GPIO.BCM)
    # for led in LED_NUMBER:
    #     GPIO.setup(led,GPIO.OUT)
    # GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    # GPIO.add_event_detect(21, GPIO.RISING, callback=button_callback, bouncetime=300)
    # print(sd.query_devices())

   # 최근 30프레임 평균
    t_prev = time.time()

    fid_log=0
    disp_fps = 0.0
    cur_fps = 0.0

    tools_result = [] 
    hand_result = None
    
    try:
        while True:
            t0 = time.time()
            # ret, frame = cap.read()
            # if not ret: break

            # ======= 추가 ========
            # while True 루프 내부 수정          
            # frames = pipeline.wait_for_frames()
            frames = pipeline.poll_for_frames()

            if not frames:
                continue


            # aligned_frames = align.process(frames) # 정렬된 프로세스 실행
            # t1 = time.time() # 뎁스 프레임 수신 및 정렬

             # ===== 3/11 필터 ====== 
            aligned_frames = align.process(frames)

            temp_color_frame = aligned_frames.get_color_frame()
            temp_depth_frame = aligned_frames.get_depth_frame()

            if not temp_color_frame or not temp_depth_frame:
                continue

            filtered_frame = threshold_filter.process(temp_depth_frame)
            filtered_frame = spatial.process(filtered_frame)
            filtered_frame = temporal.process(filtered_frame)
            # filtered_frame = hole_filling.process(filtered_frame)

            depth_frame = filtered_frame.as_depth_frame()
            color_frame = temp_color_frame
            # =========================


            # 리얼센스 데이터를 넘파이 배열로 변환 (기존 코드의 frame 역할)
            frame = np.asanyarray(color_frame.get_data())
            # ======= 추가 ========

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            annotated_image = frame.copy()

            fid += 1
            # ===== 3/16 사람인식 =====
            if current_state != 1:
                status_text = "SCANNING..."
                text_color = (255, 255, 255)
                if fid % 5 == 0:
                    # print(f">>> 사람 감지됨")
                    yolo_results = yolo_model.track(source=frame, persist=True, max_det=1, verbose=False)

                    if len(yolo_results[0]) > 0 and yolo_results[0].keypoints is not None:
                        kp_data = yolo_results[0].keypoints.xy[0]
                        valid_x = kp_data[kp_data[:, 0] > 0][:, 0]

                        if len(valid_x) > 0:
                            center_x = np.mean(valid_x.cpu().numpy() if hasattr(valid_x, 'cpu') else valid_x)
                            # 어깨/몸 중심 계산 후 구역 업데이트
                            # center_x 계산 로직...
                            if center_x < 213: target_zone = 1 # 도마
                            elif center_x < 426: target_zone = 0 # 대기
                            else: target_zone = 2              # 스토브

                            print(f"DEBUG | X: {center_x:.1f} | Zone: {target_zone} | Pending: {pending_zone}")
                            status_text = f"DETECTING - Zone: {target_zone}"
                            color = (0, 255, 0)
                            
                        
                            if target_zone != last_sent_zone:
                                if target_zone != pending_zone:
                                    pending_zone = target_zone
                                    zone_entry_time = time.time()
                                    print(f"--- 구역 변경 감지: {target_zone} (안정화 대기중...) ---")
                                elif time.time() - zone_entry_time >= STABILIZE_DELAY:
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
                cv2.putText(annotated_image, status_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)


            # [상태 1: 도마 안전 모드]
            elif current_state == 1:
                # 1. 조리 종료 시그널 확인 (예: 소켓 수신 메시지)
                if recvFlag and "FINISH" in rsplit:
                    current_state = 0
                    last_sent_zone = -1 # 구역 재탐색을 위해 초기화
                    recvFlag = False
                    print(">>> 조리 종료: 다시 사람 추적을 시작합니다.")
                    continue
                # print(f"도마모드 진입")

                # 2. 정밀 인식 (YOLO 없이 MediaPipe + ncnn만 실행)
                # 이 구역에서는 YOLO 관련 코드가 아예 실행되지 않으므로 매우 빠릅니다.
                if fid % (FRAME_SKIP + 1) == 1:                   
                    ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                    hand_result = hand_detector.detect_for_video(mp_image, ts_ms)
                if fid % 4 == 1:
                    tools_result = tools_inference(frame_rgb)
        
                if blade_pixel and fid-blade_pixel_ttl>5 :
                    blade_pixel=[]
                if target_pixel and fid-target_pixel_ttl>20 :
                    target_pixel=[]

                if hand_result is not None:
                    drawing_image = draw_landmarks_on_image(tools_result, hand_result, frame)
                    annotated_image = detection_box(tools_result, hand_result, drawing_image, depth_frame, intr)
                else: 
                    annotated_iamge = frame.copy()


                
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

            cv2.imshow("hand_land", annotated_image)


            # ======3/15 사람인식======
            # if fid % 5 == 0:
            #     yolo_results = yolo_model.track(source=frame, persist=True, max_det=1, verbose=False)
            #     if len(yolo_results[0]) > 0 and yolo_results[0].keypoints is not None:
            #         kp_data = yolo_results[0].keypoints.xy[0]
            #         valid_x = kp_data[kp_data[:, 0] > 0][:, 0]

            #         if len(valid_x) > 0:
            #             center_x = np.mean(valid_x.cpu().numpy() if hasattr(valid_x, 'cpu') else valid_x)
            #             # 어깨/몸 중심 계산 후 구역 업데이트
            #             # center_x 계산 로직...
            #             if center_x < 213: target_zone = 1  # 스토브
            #             elif center_x < 426: target_zone = 0 # 대기
            #             else: target_zone = 2               # 도마
                    
            #             if target_zone != last_sent_zone:
            #                 if target_zone != pending_zone:
            #                     pending_zone = target_zone
            #                     zone_entry_time = time.time()
            #                 elif time.time() - zone_entry_time >= STABILIZE_DELAY:
            #                     try:
            #                         s.send(f"[OMXA]MOVE@{target_zone}\n".encode())
            #                         last_sent_zone = target_zone
            #                         current_state = target_zone
            #                         print(f">>> 로봇 이동 명령 전송 : {target_zone}")
            #                     except: pass




            # # 도마 구역일때
            # if current_state == 1:
            #     # ======3/9수정=======
            #     # 4프레임마다 도구 인식(도구 인식 유지하는 코드있으므로) 

            #     if fid % (FRAME_SKIP + 1) == 1:
            #         # tools_result = tools_inference(frame_rgb)
            #         t2 = time.time() # ncnn 모델 추론
            #         ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            #         hand_result = hand_detector.detect_for_video(mp_image, ts_ms)
            #         t3 = time.time() # mediapipe 손 인식

            #     if fid % 4 == 1:
            #         tools_result = tools_inference(frame_rgb)

            #     drawing_image = draw_landmarks_on_image(tools_result, hand_result, frame)
            #     # annotated_image, audio_event = detection_box(tools_result, hand_result, drawing_image)
            #     annotated_image = detection_box(tools_result, hand_result, drawing_image, depth_frame, intr)
            #     t4 = time.time() # 후처리 및 그리기

            #     t_now = time.time()
            #     elapsed_time = t_now - t_prev
            #     if elapsed_time > 1.0: # 1초가 지날 때마다
            #         cur_fps = (fid - fid_log) / elapsed_time # (현재 프레임 - 1초 전 프레임) / 시간
            #         fid_log = fid
            #         t_prev = t_now

            #     # 6. 화면에 FPS 그리기
            #     # annotated_image 상단에 노란색으로 표시합니다.
            #     h, w, _ = annotated_image.shape
            #     cv2.putText(annotated_image, f"FPS: {cur_fps:.1f}", (w - 150, 30), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            #     if blade_pixel and fid-blade_pixel_ttl>5 :
            #         blade_pixel=[]
            #     if target_pixel and fid-target_pixel_ttl>20 :
            #         target_pixel=[]
            #     t_now = time.time()
            #     elased_time=t_now-t_prev
            #     # ==================
            # else:
            #     # [대기/스토브 모드] YOLO 뼈대만 출력하거나 기본 화면 유지
            #     # annotated_image가 항상 정의되도록 보장
            #     annotated_image = frame.copy() 
            #     if 'yolo_results' in locals() and len(yolo_results[0]) > 0:
            #         annotated_image = yolo_results[0].plot() # 사람 뼈대 그리기
                
            #     mode_name = "STOVE" if current_state == 2 else "STANDBY"
            #     cv2.putText(annotated_frame if 'annotated_frame' in locals() else annotated_image, 
            #                 f"STATUS: {mode_name} MODE", (30, 100), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # =========================

                
            



            # 터미널에 각 구간 소요 시간 출력 (단위: ms)
            # if fid % 30 == 0: # 30프레임마다 한 번씩 출력
            #     print(f"Align: {(t1-t0)*1000:.1f}ms | Tools: {(t2-t1)*1000:.1f}ms | Hand: {(t3-t2)*1000:.1f}ms | Box: {(t4-t3)*1000:.1f}ms")

            # if blade_pixel and fid-blade_pixel_ttl>5 and audio_event=='None':
            #     blade_pixel=[]
            # if target_pixel and fid-target_pixel_ttl>20 and audio_event=='None':
            #     target_pixel=[]

            # if elased_time>1.0:
            #     print(audio_event=="detected")
            #     t_prev+=elased_time
            # if audio_event == "danger":
            #     set_hold_mode("danger")
            # elif audio_event == "detected":
            #     set_hold_mode("detected")
            # else:
            #     set_hold_mode(None)

            # if hand_result.hand_landmarks:
            #     for hand_landmarks in hand_result.hand_landmarks:
            #         finger_tips = hand_landmarks[8]
            #         u_h = int(np.clip(finger_tips.x * 640, 0, 639))
            #         v_h = int(np.clip(finger_tips.y * 480, 0, 479))

            #         for obj in tools_result:
            #             x1,y1,x2,y2 = obj[0]
            #             obj_y = int ((x1+x2)/2)
                        


            # === FPS 갱신 ===
            '''
            t_now = time.time()
            elased_time=t_now-t_prev
            if elased_time>1.0:
                #print(f"fps:{((fid-fid_log)/elased_time):5.2f}")
                fid_log=fid
                t_prev+=elased_time
            '''
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    # cap.release()
    
    # cv2.waitKey(1)
    # GPIO.cleanup()

if __name__ == "__main__":
    main()