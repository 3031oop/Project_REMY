#aruco_client

다른 client[ex: WF]에서
[ARUCO]id1 보냄 -> 뎁스카메라로 id1 aruco marker인식 -> [WF]X@Y@Z 메세지보냄
[ARUCO]id2 보냄 -> 뎁스카메라로 id1 aruco marker인식 -> [WF]X@Y@Z 메세지보냄

#voice_audio_node_0318.py 
윈도우에서 실행해야함
C:\Users\KCCISTC\OneDrive\Desktop\굿> python voice_audio_node_0318.py
wav파일은 notion에 존재

#bridge node실행
ros2 run waffle_bridge waffle_bridge_node --ros-args -p server_ip:='10.10.141.50' -p server_port:=5000 -p password:='PASSWD'

#waffle yolo 노드
OPENBLAS_CORETYPE=ARMV8 ros2 run hsj_waffle_py yolo_node
