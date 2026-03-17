#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <set>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class WaffleBridgeClient : public rclcpp::Node
{
public:
  WaffleBridgeClient()
  : Node("waffle_bridge_client"),
    sock_fd_(-1),
    connected_(false),
    stop_recv_(false)
  {
    // 1) parameter setup
    server_ip_ = this->declare_parameter<std::string>("server_ip", "10.10.141.50");
    server_port_ = this->declare_parameter<int>("server_port", 5000);
    client_id_ = this->declare_parameter<std::string>("client_id", "HSJ_WF");
    password_ = this->declare_parameter<std::string>("password", "PASSWD");
    reconnect_period_ms_ = this->declare_parameter<int>("reconnect_period_ms", 2000);

    allowed_sender_ = {"VOI", "ADMIN", "HSJ_AND"};

    RCLCPP_INFO(this->get_logger(),
                "Connecting to %s:%d (ID:%s, PW:%s)",
                server_ip_.c_str(),
                server_port_,
                client_id_.c_str(),
                password_.c_str());

    // 2) ROS 2 Publisher & Subscription
    // 명령 입력 토픽
    wf_command_pub_ = this->create_publisher<std_msgs::msg::String>("/waffle_command", 10);

    // 상태 출력 토픽
    wf_status_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/waffle_status",
      10,
      std::bind(&WaffleBridgeClient::statusCallback, this, std::placeholders::_1));

    // 3) reconnect timer
    reconnect_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(reconnect_period_ms_),
      std::bind(&WaffleBridgeClient::reconnectIfNeeded, this));

    connectToServer();
  }

  ~WaffleBridgeClient() override
  {
    stop_recv_ = true;
    connected_ = false;
    closeSocket();
    if (recv_thread_.joinable()) {
      recv_thread_.join();
    }
  }

private:
  // =========================
  // ROS status -> TCP
  // yolo_state_node가 /waffle_status 에 publish하는 상태를 서버로 중계
  // =========================
  void statusCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!connected_) {
      RCLCPP_WARN(this->get_logger(), "server not connected - status send canceled");
      return;
    }

    const std::string & status = msg->data;

    // 팀 요구사항 기준 허용 상태
    static const std::set<std::string> allowed_status = {
      "patrol",
      "detect",
      "depart",
      "wait_return",
      "force_return",
      "return",
      "idle"
    };

    if (allowed_status.find(status) == allowed_status.end()) {
      RCLCPP_WARN(this->get_logger(), "ignored unknown waffle status: %s", status.c_str());
      return;
    }

    // 현재 서버 형식 호환: [ALLMSG]status
    std::string wire_msg = "[ALLMSG]" + status + "\n";

    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (sock_fd_ >= 0) {
      ::write(sock_fd_, wire_msg.c_str(), wire_msg.size());
      RCLCPP_INFO(this->get_logger(), "server report success: %s", wire_msg.c_str());
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
    if (connected_) return;

    closeSocketUnlocked();

    sock_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ < 0) return;

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(static_cast<uint16_t>(server_port_));
    inet_pton(AF_INET, server_ip_.c_str(), &serv_addr.sin_addr);

    if (::connect(sock_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      closeSocketUnlocked();
      return;
    }

    // 서버 인증: [HSJ_WF:PASSWD]
    std::string auth_msg = "[" + client_id_ + ":" + password_ + "]";
    ::write(sock_fd_, auth_msg.c_str(), auth_msg.size());

    connected_ = true;
    RCLCPP_INFO(this->get_logger(), "TCP server connect success (%s:%d)", server_ip_.c_str(), server_port_);
    RCLCPP_INFO(this->get_logger(), "server auth sent: %s", auth_msg.c_str());

    if (!recv_thread_.joinable()) {
      stop_recv_ = false;
      recv_thread_ = std::thread(&WaffleBridgeClient::recvLoop, this);
    }
  }

  // =========================
  // TCP -> ROS command
  // 서버에서 [VOI]tts1 같은 메시지를 받으면
  // /waffle_command 로 start/return/force_return publish
  // =========================
  void recvLoop()
  {
    char buffer[512];
    std::string accum;

    while (!stop_recv_ && rclcpp::ok()) {
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
      ssize_t len = ::read(local_sock, buffer, sizeof(buffer) - 1);

      if (len <= 0) {
        RCLCPP_WARN(this->get_logger(), "server disconnected!");
        connected_ = false;
        closeSocket();
        continue;
      }

      accum += std::string(buffer, len);

      while (true) {
        auto pos = accum.find('\n');
        if (pos == std::string::npos) break;

        std::string line = accum.substr(0, pos);
        accum.erase(0, pos + 1);

        trim(line);
        if (line.empty()) continue;

        handleServerMessage(line);
      }
    }
  }

  void handleServerMessage(const std::string & received_raw)
  {
    auto msg = std_msgs::msg::String();

    size_t open_bracket = received_raw.find('[');
    size_t close_bracket = received_raw.find(']');

    if (open_bracket == std::string::npos || close_bracket == std::string::npos) {
      RCLCPP_WARN(this->get_logger(), "Malformed msg: %s", received_raw.c_str());
      return;
    }

    std::string sender_id = received_raw.substr(open_bracket + 1, close_bracket - open_bracket - 1);
    std::string pure_cmd = received_raw.substr(close_bracket + 1);

    trim(sender_id);
    trim(pure_cmd);

    if (allowed_sender_.find(sender_id) == allowed_sender_.end()) {
      RCLCPP_WARN(this->get_logger(), "Unauthorized sender (%s) - ignored", sender_id.c_str());
      return;
    }

    // 현재 팀 규약 유지
    if (pure_cmd == "tts1") {
      msg.data = "start";
      RCLCPP_INFO(this->get_logger(), "TCP '%s' from %s -> ROS 'start'", pure_cmd.c_str(), sender_id.c_str());
    } else if (pure_cmd == "tts2") {
      msg.data = "return";
      RCLCPP_INFO(this->get_logger(), "TCP '%s' from %s -> ROS 'return'", pure_cmd.c_str(), sender_id.c_str());
    } else if (pure_cmd == "tts3") {
      msg.data = "force_return";
      RCLCPP_INFO(this->get_logger(), "TCP '%s' from %s -> ROS 'force_return'", pure_cmd.c_str(), sender_id.c_str());
    }
    // 추후 확장용 direct command
    else if (pure_cmd == "start") {
      msg.data = "start";
    } else if (pure_cmd == "return") {
      msg.data = "return";
    } else if (pure_cmd == "force_return") {
      msg.data = "force_return";
    } else {
      RCLCPP_WARN(this->get_logger(), "Unknown command payload: %s", pure_cmd.c_str());
      return;
    }

    wf_command_pub_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Published /waffle_command: %s", msg.data.c_str());
  }

  static void trim(std::string & s)
  {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r' || s.front() == '\t')) {
      s.erase(s.begin());
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) {
      s.pop_back();
    }
  }

  void closeSocket()
  {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    closeSocketUnlocked();
  }

  void closeSocketUnlocked()
  {
    if (sock_fd_ >= 0) {
      ::close(sock_fd_);
      sock_fd_ = -1;
    }
  }

private:
  std::string server_ip_;
  std::string client_id_;
  std::string password_;
  int server_port_;
  int reconnect_period_ms_;

  std::set<std::string> allowed_sender_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr wf_command_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr wf_status_sub_;
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
  auto node = std::make_shared<WaffleBridgeClient>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
