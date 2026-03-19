import cv2
import numpy as np
import pyrealsense2 as rs
import socket
import threading

# --- IoT 서버 설정 ---
SERVER_IP = '127.0.0.1'
SERVER_PORT = 5000
MY_ID = 'ARUCO'
MY_PW = 'PASSWD'

# 전역 변수
current_target_id = None 
last_sender = "ALLMSG"  # 마지막으로 명령을 내린 클라이언트 ID 저장

def recv_thread(sock):
    global current_target_id, last_sender
    try:
        while True:
            data = sock.recv(1024).decode('utf-8')
            if not data: break
            
            # 서버 메시지 파싱: "[KSH]id1" -> sender: "KSH", content: "id1"
            if ']' in data:
                parts = data.split(']')
                sender = parts[0].replace('[', '') # "KSH"
                msg_content = parts[1].strip().lower() # "id1"
                
                if msg_content.startswith("id"):
                    try:
                        new_id = int(msg_content.replace("id", ""))
                        current_target_id = new_id
                        last_sender = sender  # 👉 명령을 내린 사람 기억 (나중에 좌표 쏠 곳)
                        print(f"🎯 타겟 변경: ID {new_id} (명령자: {last_sender})")
                    except ValueError:
                        pass
                elif msg_content == "stop":
                    current_target_id = None
                    print("🛑 추적 중지")
    except Exception as e:
        print(f"❌ 수신 오류: {e}")

def main():
    global current_target_id, last_sender

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((SERVER_IP, SERVER_PORT))
        sock.send(f"[{MY_ID}:{MY_PW}]".encode())
    except:
        return

    threading.Thread(target=recv_thread, args=(sock,), daemon=True).start()

    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipe.start(config)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            corners, ids, _ = detector.detectMarkers(color_image)

            if current_target_id is not None and ids is not None:
                intr = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

                for i in range(len(ids)):
                    if int(ids[i][0]) == current_target_id:
                        u = int(np.mean(corners[i][0][:, 0]))
                        v = int(np.mean(corners[i][0][:, 1]))
                        z = depth_frame.get_distance(u, v)

                        if z > 0:
                            x, y, z = rs.rs2_deproject_pixel_to_point(intr, [u, v], z)
                            
                            # 👉 [명령자ID]X@Y@Z 형식으로 서버에 전송
                            # 예: [KSH]0.123@-0.045@0.850
                            response_msg = f"[{last_sender}]{x:.3f}@{y:.3f}@{z:.3f}\n"
                            sock.send(response_msg.encode())
                            
                            # 로컬 확인용
                            print(f"📤 전송 완료: {response_msg.strip()}")

            cv2.imshow("IoT ArUco Client", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        pipe.stop()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
