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
import math
import time

# =====3/11 소켓 추가 ======
import socket
import threading
import re

HOST = "127.0.0.1"
# HOST = "10.10.141.43" 
PORT = 5000
# TARGET_ID = "[REMY_OMXA]"

recvFlag = False
rsplit = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def gettingMsg(): 
    global rsplit, recvFlag, TARGET, target_pixel
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
net.load_param("kitchen_tools_best03_50.ncnn.param")
net.load_model("kitchen_tools_best03_50.ncnn.bin")
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

# ===== 3/12 좌표 ======
prev_blade_pos = [0, 0, 0] # 이전 좌표 저장
MOVE_THRESHOLD = 5.0  # 5mm 이상 움직여야 전송
# =====================



# ===== 그리기 =====
def draw_landmarks_on_image(tools_dect, hand_dect, frame):
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
def inside_allhand(hand_pixel,target_pixel,W,H,depth_frame,is_blade=False):
    global last_blade_z
    min_dz = 99.9

    if is_blade and len(target_pixel) > 0:
        target = target_pixel[0]
        tx, ty = (target[0]+ target[2]) // 2, (target[1]+ target[3]) // 2
   
        current_z = depth_frame.get_distance(tx,ty)
        if current_z > 0:
            last_blade_z = current_z

    for lm in hand_pixel:
        lm_x, lm_y = int(lm.x * W), int(lm.y * H)

        for target in target_pixel:
            if check_inside(target, lm_x,lm_y):
                if is_blade:
                    dist_hand = depth_frame.get_distance(lm_x,lm_y)
                    if dist_hand > 0 and last_blade_z > 0:
                        dz = abs(dist_hand - last_blade_z)
                        if dz < min_dz: min_dz = dz # 가장 가까운 마디의 거리 저장
                        if dz < 0.02:
                            return True, dz # 충돌 시 현재 거리 반환
                else:
                    return True, 0.0
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
    target_id = "[REMY_VI]"

    # ====== 3/12 좌표 =======
    global prev_blade_pos 
    pos_id = "[REMY_OMXB]"
    # =======================


    blade_xy  = [ tools[0] for tools in tools_dect if tools[2] == 0 ]
    # setting_target(tools_dect)
    setting_target(tools_result=tools_dect)

    global target_pixel, blade_pixel,flag_target,flag_blade,blade_pixel_ttl,fid
    if len(blade_xy)  > 0:
        blade_pixel  = save_pixel(blade_xy)
        blade_pixel_ttl=fid
    annotated_image = np.copy(current_frame)

    flag_Up = False; flag_Down = False; flag_Left = False; flag_Right = False

    hand_list = hand_dect.hand_landmarks
    H, W, _ = annotated_image.shape
    
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

    flag_target, _= inside_allhand(hand_pixel,target_pixel,W,H,depth_frame, False)
    flag_blade, dz = inside_allhand(hand_pixel,blade_pixel,W,H, depth_frame,True)

    for obj in tools_dect:
        (x1, y1, x2, y2), score, class_id = obj
        
        if class_id == 0:  # 칼날(Blade)인 경우
            # 1. 칼날의 중심 픽셀 구하기
            bx, by = (x1 + x2) // 2, (y1 + y2) // 2
            bx, by = np.clip(bx, 0, W-1), np.clip(by, 0, H-1)
            
            # 2. 해당 지점의 깊이(Z) 값 읽기
            dist_b = depth_frame.get_distance(bx, by)

            if dist_b > 0:
                
                # 3. 픽셀 좌표를 카메라 기준 실제 3D 좌표(m)로 변환
                # p_blade는 [X, Y, Z] 형태의 리스트가 됩니다.

                # p_blade = rs.rs2_deproject_pixel_to_point(intr, [bx, by], dist_b)
                
                # blade_x = p_blade[0] * 1000
                # blade_y = p_blade[1] * 1000
                # blade_z = p_blade[2] * 1000


                # ====== 3/12 좌표 ======
                p_blade_m = rs.rs2_deproject_pixel_to_point(intr, [bx,by], dist_b)
                curr_x, curr_y, curr_z = [p * 1000 for p in p_blade_m] # mm 변환

                # 이전 좌표와의 직선 거리 계산 (피타고라스)
                dist_moved = math.sqrt((curr_x - prev_blade_pos[0])**2 +
                                        (curr_y - prev_blade_pos[1])**2 +
                                        (curr_z - prev_blade_pos[2])**2)

              
                if dist_moved >MOVE_THRESHOLD:
                    try:
                        pos_msg = f"{pos_id}{curr_x:.0f},{curr_y:.0f},{curr_z:.0f}\n"
                        s.send(pos_msg.encode())
                        print(f"{curr_x:.0f},{curr_y:.0f},{curr_z:.0f}")
                        prev_blade_pos = [curr_x, curr_y, curr_z]

                    except:
                        pass
                # ======================



                # # 4. 소켓 전송 (칼의 ID와 좌표 전송)
                # # 포맷: "BLADE_POS,ID,X,Y,Z"
                # socket_msg = f"BLADE_POS,0,{blade_x:.3f},{blade_y:.3f},{blade_z:.3f}\n"
                # # client_socket.sendall(socket_msg.encode())

                # 화면 디버깅 출력
                # 칼날의 위치 출력 (감지 박스 안의 중앙 지점)
                # cv2.putText(annotated_image, f"Blade XYZ: {blade_x:.2f}, {blade_y:.2f}, {blade_z:.2f}", 
                #             (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # print(f"Blade XYZ: {blade_x:.2f}, {blade_y:.2f}, {blade_z:.2f}")

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
        if flag_blade:
            draw_text(annotated_image, "DANGER! ", WARNING_TXT_ORG, COLOR_RED)
            s.send(f"{target_id}DANGER\n".encode()) # 위험 신호 전송
            print(f"danger")
        elif flag_target:
            draw_text(annotated_image, "DETECTED!", WARNING_TXT_ORG, COLOR_GREEN)
            s.send(f"{target_id}DETECTED\n".encode()) # 감지 신호 전송
            print(f"detected")

        if dz <10.0:
            debug_text = f"Z-dist: {dz:.3f}m"
            cv2.putText(annotated_image, debug_text, (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # print(f"[DEBUG] 현재 손-칼날 거리: {dz:.3f}m")

        if flag_Left:
            if flag_Up:    s.send(f"{target_id}LED@22\n".encode())
            elif flag_Down:s.send(f"{target_id}LED@17\n".encode())
            else:          s.send(f"{target_id}LED@27\n".encode())
        elif flag_Right:
            if flag_Up:    s.send(f"{target_id}LED@4\n".encode())
            elif flag_Down:s.send(f"{target_id}LED@2\n".encode())
            else:          s.send(f"{target_id}LED@3\n".encode())
        # print(f"L:{flag_Left} R:{flag_Right} U:{flag_Up} D:{flag_Down}")
    except Exception as e:
        # 전송 실패 시 조용히 넘어감 (서버 연결 끊김 대비)
        pass


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

def tools_inference(frame_rgb):
    H, W, _ = frame_rgb.shape            # (버그 수정) frame -> frame_rgb
    img_lbx, r, lpad, tpad = letterbox(frame_rgb, IMG_SIZE)
    # (버그 수정) img_rgb → img_lbx는 이미 RGB
    input_mat = ncnn.Mat.from_pixels(img_lbx, ncnn.Mat.PixelType.PIXEL_RGB, IMG_SIZE, IMG_SIZE)
    input_mat.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])

    ex = net.create_extractor()
    ex.input("in0", input_mat)
    _, result = ex.extract("out0")

    arr = result.numpy()
    D, N = arr.shape
    A = arr.T
    
    cx, cy, w, h = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    cls_scores = A[:, 4:]
    
    cls_ids = np.argmax(cls_scores, axis=1)
    scores  = cls_scores[np.arange(A.shape[0]), cls_ids]
    
    keep = scores >= CONF_TH
    if not np.any(keep): return []
    cx = cx[keep]; cy = cy[keep]; w = w[keep]; h = h[keep]
    scores = scores[keep]; cls_ids = cls_ids[keep]

    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    x1 = (x1 - lpad) / r; y1 = (y1 - tpad) / r
    x2 = (x2 - lpad) / r; y2 = (y2 - tpad) / r

    x1 = np.clip(x1, 0, W); y1 = np.clip(y1, 0, H)
    x2 = np.clip(x2, 0, W); y2 = np.clip(y2, 0, H)

    dets = [([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
             float(scores[i]), int(cls_ids[i])) for i in range(len(scores))]
    dets = nms(dets, IOU_TH)
    return dets

# ===== 메인 =====
def main():
    try:
        s.connect((HOST, PORT))
        s.send('[REMY_OMXA:1234]'.encode())
        threading.Thread(target=gettingMsg, daemon=True).start()
        print(">>> 서버 연결 성공!")
    except: print(">>> 서버 연결 실패 (통신 제외 실행)")
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

    global fid,blade_pixel_ttl,blade_pixel,target_pixel,target_pixel_ttl

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


            aligned_frames = align.process(frames) # 정렬된 프로세스 실행
            t1 = time.time() # 뎁스 프레임 수신 및 정렬

            # 정렬된 프레임 꺼내기
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 리얼센스 데이터를 넘파이 배열로 변환 (기존 코드의 frame 역할)
            frame = np.asanyarray(color_frame.get_data())
            # ======= 추가 ========

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            
            # ======3/9수정=======
            # 4프레임마다 도구 인식(도구 인식 유지하는 코드있으므로) 
            fid += 1
            if fid % (FRAME_SKIP + 1) == 1:
                # tools_result = tools_inference(frame_rgb)
                t2 = time.time() # ncnn 모델 추론
                ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                hand_result = hand_detector.detect_for_video(mp_image, ts_ms)
                t3 = time.time() # mediapipe 손 인식

            if fid % 4 == 1:
                tools_result = tools_inference(frame_rgb)
            # ==================
            
            drawing_image = draw_landmarks_on_image(tools_result, hand_result, frame)
            # annotated_image, audio_event = detection_box(tools_result, hand_result, drawing_image)
            annotated_image = detection_box(tools_result, hand_result, drawing_image, depth_frame, intr)
            t4 = time.time() # 후처리 및 그리기

            t_now = time.time()
            elapsed_time = t_now - t_prev
            if elapsed_time > 1.0: # 1초가 지날 때마다
                cur_fps = (fid - fid_log) / elapsed_time # (현재 프레임 - 1초 전 프레임) / 시간
                fid_log = fid
                t_prev = t_now

            # 6. 화면에 FPS 그리기
            # annotated_image 상단에 노란색으로 표시합니다.
            h, w, _ = annotated_image.shape
            cv2.putText(annotated_image, f"FPS: {cur_fps:.1f}", (w - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            # 터미널에 각 구간 소요 시간 출력 (단위: ms)
            # if fid % 30 == 0: # 30프레임마다 한 번씩 출력
            #     print(f"Align: {(t1-t0)*1000:.1f}ms | Tools: {(t2-t1)*1000:.1f}ms | Hand: {(t3-t2)*1000:.1f}ms | Box: {(t4-t3)*1000:.1f}ms")

            # if blade_pixel and fid-blade_pixel_ttl>5 and audio_event=='None':
            #     blade_pixel=[]
            # if target_pixel and fid-target_pixel_ttl>20 and audio_event=='None':
            #     target_pixel=[]
            if blade_pixel and fid-blade_pixel_ttl>5 :
                blade_pixel=[]
            if target_pixel and fid-target_pixel_ttl>20 :
                target_pixel=[]
            t_now = time.time()
            elased_time=t_now-t_prev
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
                        

            cv2.imshow("hand_land", annotated_image)
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

