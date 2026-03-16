#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <sstream>
#include <vector>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

using namespace std::chrono_literals;

class TcpBridgeClient : public rclcpp::Node
{
public:
  TcpBridgeClient() : Node("ros2_tcp_bridge_client"), sock_fd_(-1), connected_(false), stop_recv_(false)
  {
    server_ip_ = this->declare_parameter<std::string>("server_ip", "127.0.0.1");
    server_port_ = this->declare_parameter<int>("server_port", 5000);
    client_id_ = this->declare_parameter<std::string>("client_id", "OMXA");
    password_ = this->declare_parameter<std::string>("password", "PASSWD");
    target_id_ = this->declare_parameter<std::string>("target_id", "ARD");
    reconnect_period_ms_ = this->declare_parameter<int>("reconnect_period_ms", 2000);

    target_pose_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/target_pose", 10);
    arm_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("/arm_controller/joint_trajectory", 10);

    omx_command_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/omx_command", 10,
      std::bind(&TcpBridgeClient::omxCallback, this, std::placeholders::_1));

    reconnect_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(reconnect_period_ms_),
      std::bind(&TcpBridgeClient::reconnectIfNeeded, this));

    connectToServer();
  }

  ~TcpBridgeClient() override
  {
    stop_recv_ = true;
    connected_ = false;
    closeSocket();
    if (recv_thread_.joinable()) {
      recv_thread_.join();
    }
  }

private:
  void omxCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    const std::string wire_msg = "[" + target_id_ + "]" + msg->data;

    if (!connected_) {
      return;
    }

    if (!sendRaw(wire_msg)) {
      connected_ = false;
      closeSocket();
    } 
    else {
      RCLCPP_INFO(this->get_logger(), "OMX: %s", wire_msg.c_str());
    }
  }

  void reconnectIfNeeded()
  {
    if (!connected_) {
      connectToServer();
    }
  }

  void connectToServer()
  {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    if (connected_) {
      return;
    }

    closeSocketUnlocked();

    sock_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
      return;
    }

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(static_cast<uint16_t>(server_port_));

    if (::inet_pton(AF_INET, server_ip_.c_str(), &serv_addr.sin_addr) <= 0) {
      closeSocketUnlocked();
      return;
    }

    if (::connect(sock_fd_, reinterpret_cast<sockaddr *>(&serv_addr), sizeof(serv_addr)) < 0) {
      closeSocketUnlocked();
      return;
    }

    const std::string auth_msg = "[" + client_id_ + ":" + password_ + "]";
    if (!sendRawUnlocked(auth_msg)) {
      closeSocketUnlocked();
      return;
    }

    connected_ = true;
    RCLCPP_INFO(this->get_logger(), "Connected to %s:%d", server_ip_.c_str(), server_port_);

    if (!recv_thread_.joinable()) {
      stop_recv_ = false;
      recv_thread_ = std::thread(&TcpBridgeClient::recvLoop, this);
    }
  }

  void recvLoop()
  {
    constexpr size_t kBufferSize = 256;
    char buffer[kBufferSize];

    while (!stop_recv_) {
      int local_sock = -1;
      {
        std::lock_guard<std::mutex> lock(socket_mutex_);
        local_sock = sock_fd_;
      }

      if (!connected_ || local_sock < 0) {
        std::this_thread::sleep_for(200ms);
        continue;
      }

      std::memset(buffer, 0, sizeof(buffer));
      const ssize_t read_len = ::read(local_sock, buffer, sizeof(buffer) - 1);
      if (read_len <= 0) {
        connected_ = false;
        closeSocket();
        std::this_thread::sleep_for(500ms);
        continue;
      }

      buffer[read_len] = '\0';
      std::string raw_msg(buffer);
      
      while (!raw_msg.empty() && (raw_msg.back() == '\n' || raw_msg.back() == '\r' || raw_msg.back() == ' ')) {
        raw_msg.pop_back();
      }
      RCLCPP_INFO(this->get_logger(), "Received: %s", raw_msg.c_str());

      // [ID]payload 형식 확인
      size_t lbracket = raw_msg.find('[');
      size_t rbracket = raw_msg.find(']');

      if (lbracket != 0 || rbracket == std::string::npos || rbracket <= 1) {
        RCLCPP_WARN(this->get_logger(), "Invalid message format: %s", raw_msg.c_str());
        continue;
      }

      // [ID] 분리
      std::string recv_id = raw_msg.substr(1, rbracket - 1);
      std::string payload = raw_msg.substr(rbracket + 1);

      if (payload.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty payload: %s", raw_msg.c_str());
        continue;
      }

      if (payload.find('@') != std::string::npos) {
        std::vector<std::string> cmd_tokens = splitString(payload, '@');

        if (cmd_tokens.size() < 2) {
          RCLCPP_WARN(this->get_logger(), "Invalid command payload: %s", payload.c_str());
          continue;
        }

        const std::string & command = cmd_tokens[0];
        const std::string & argument = cmd_tokens[1];

        // 여기서 명령별 처리
        if (command == "MOVE") {
          if (argument == "0") {
            publishArmMoveCommand(0);
          } 
          else if (argument == "1") {
            publishArmMoveCommand(1);
          } 
          else if (argument == "2") {
            publishArmMoveCommand(2);
          }
        }
        continue;
      }

      if (payload.find(',') != std::string::npos) {
        std::vector<std::string> value_tokens = splitString(payload, ',');

        if (!isFloatList(value_tokens)) {
          RCLCPP_WARN(this->get_logger(), "Invalid float payload: %s", payload.c_str());
          continue;
        }

        std_msgs::msg::Float32MultiArray out_msg;

        for (const auto & token : value_tokens) {
          out_msg.data.push_back(std::stof(token));
        }

        target_pose_pub_->publish(out_msg);

        RCLCPP_INFO(this->get_logger(), "POSE RX -> published %zu values", out_msg.data.size());

        continue;
      }
    }
  }

  bool sendRaw(const std::string & msg)
  {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    return sendRawUnlocked(msg);
  }

  bool sendRawUnlocked(const std::string & msg)
  {
    if (sock_fd_ < 0) {
      return false;
    }

    const ssize_t written = ::write(sock_fd_, msg.c_str(), msg.size());
    return written == static_cast<ssize_t>(msg.size());
  }

  void closeSocket()
  {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    closeSocketUnlocked();
  }

  void closeSocketUnlocked()
  {
    if (sock_fd_ >= 0) {
      ::shutdown(sock_fd_, SHUT_RDWR);
      ::close(sock_fd_);
      sock_fd_ = -1;
    }
  }

  std::vector<std::string> splitString(const std::string & input, char delim)
  {
    std::vector<std::string> tokens;
    std::stringstream ss(input);
    std::string item;

    while (std::getline(ss, item, delim)) {
      tokens.push_back(item);
    }

    return tokens;
  }

  bool isFloatList(const std::vector<std::string> & tokens)
  {
    if (tokens.empty()) {
      return false;
    }

    try {
      for (const auto & t : tokens) {
        if (t.empty()) {
          return false;
        }
        std::stof(t);
      }
    } 
    catch (const std::exception &) {
      return false;
    }

    return true;
  }

  double deg2rad(double deg)
  {
    return deg * M_PI / 180.0;
  }

  void publishArmMoveCommand(int move_index)
  {
    std::array<double, 5> joint_deg{};

    // 대기
    if (move_index == 0) {
      joint_deg = {45.0, -100.0, 60.0, 30.0, 0.0};
    } 
    // 요리중
    else if (move_index == 1) {
      joint_deg = {0.0, -30.0, -20.0, 60.0, 0.0};
    } 
    // 화구
    else if (move_index == 2) {
      joint_deg = {90.0, 0.0, -45.0, 60.0, 0.0};
    } 
    else {
      RCLCPP_WARN(this->get_logger(), "Unknown MOVE index: %d", move_index);
      return;
    }

    trajectory_msgs::msg::JointTrajectory traj_msg;
    traj_msg.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5"};

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {
      deg2rad(joint_deg[0]),
      deg2rad(joint_deg[1]),
      deg2rad(joint_deg[2]),
      deg2rad(joint_deg[3]),
      deg2rad(joint_deg[4])
    };
    point.time_from_start.sec = 2;
    point.time_from_start.nanosec = 0;

    traj_msg.points.push_back(point);

    arm_pub_->publish(traj_msg);

    RCLCPP_INFO(this->get_logger(), "Published MOVE %d -> [%.1f, %.1f, %.1f, %.1f, %.1f] deg", move_index, joint_deg[0], joint_deg[1], joint_deg[2], joint_deg[3], joint_deg[4]);
  }

  std::string server_ip_;
  int server_port_;
  std::string client_id_;
  std::string password_;
  std::string target_id_;
  std::string default_target_;
  int reconnect_period_ms_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr target_pose_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr arm_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr omx_command_sub_;
  rclcpp::TimerBase::SharedPtr reconnect_timer_;

  std::thread recv_thread_;
  std::mutex socket_mutex_;
  int sock_fd_;
  std::atomic<bool> connected_;
  std::atomic<bool> stop_recv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TcpBridgeClient>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
