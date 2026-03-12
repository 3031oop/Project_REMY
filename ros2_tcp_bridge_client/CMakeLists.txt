cmake_minimum_required(VERSION 3.8)
project(ros2_tcp_bridge_client)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(tcp_bridge_client_node src/tcp_bridge_client.cpp)
ament_target_dependencies(
  tcp_bridge_client_node 
  rclcpp 
  std_msgs
  trajectory_msgs
)
target_compile_features(tcp_bridge_client_node PUBLIC cxx_std_17)

install(TARGETS
  tcp_bridge_client_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
