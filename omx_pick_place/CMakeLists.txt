cmake_minimum_required(VERSION 3.8)
project(omx_pick_place)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ---------- srv ----------
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PoseToJointExecute.srv"
  "srv/PickPlace.srv"
  DEPENDENCIES geometry_msgs sensor_msgs
)
# ---------- executable ----------
add_executable(pick_place_node
  src/pick_place_node.cpp
)

ament_target_dependencies(pick_place_node
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  sensor_msgs
)
add_executable(pose_to_joint_execute_server src/pose_to_joint_execute_server.cpp)
ament_target_dependencies(pose_to_joint_execute_server
  rclcpp
  rclcpp_action
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  control_msgs
  tf2_ros
  tf2_geometry_msgs
  moveit_ros_planning_interface
)
# srv type support link
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(pose_to_joint_execute_server
  "${cpp_typesupport_target}"
)
target_link_libraries(pick_place_node
  "${cpp_typesupport_target}"
)
ament_export_dependencies(rosidl_default_runtime)
install(TARGETS
  pick_place_node
  pose_to_joint_execute_server
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
