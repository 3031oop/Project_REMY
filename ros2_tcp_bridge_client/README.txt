기본 실행 명령어
ros2 run ros2_tcp_bridge_client tcp_bridge_client_node

파라미터 수정 실행
ros2 run ros2_tcp_bridge_client tcp_bridge_client_node --ros-args -p client_id:=BRIDGE3 .....

아래파일들 교체
open_manipulator_description/ros2_control/omx_f.ros2_control.xacro
open_manipulaotr_description/urdf/omx_f/omx_f_arm.urdf.xacro
open_manipulator_bringup/config/omx_f/hardware_controller_manager.yaml
open_manipulator_bringup/launch/omx_f.launch.py
