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

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class WaffleBridgeClient : public rclcpp::Node
{
public:
  WaffleBridgeClient() : Node("waffle_bridge_client"), sock_fd_(-1), connected_(false), stop_recv_(false)
  {
    // 1. parameter setup
    server_ip_ = this->declare_parameter<std::string>("server_ip", "127.0.0.1");
    server_port_ = this->declare_parameter<int>("server_port", 5000);
	client_id_ = this->declare_parameter<std::string>("client_id", "HSJ_WF");
	password_ = this->declare_parameter<std::string>("password", "PASSWD");
    reconnect_period_ms_ = this->declare_parameter<int>("reconnect_period_ms", 2000);

    // 2. ROS 2 Publisher & Subscription
    wf_command_pub_ = this->create_publisher<std_msgs::msg::String>("/waffle_command", 10);
    wf_command_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/waffle_command", 10, std::bind(&WaffleBridgeClient::rosCallback, this, std::placeholders::_1));

    // 3.reconnection timer
    reconnect_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(reconnect_period_ms_),
      std::bind(&WaffleBridgeClient::reconnectIfNeeded, this));

    connectToServer();
  }

  ~WaffleBridgeClient() override {
    stop_recv_ = true;
    connected_ = false;
    closeSocket();
    if (recv_thread_.joinable()) recv_thread_.join();
  }

private:
  // [robot -> server]
  void rosCallback(const std_msgs::msg::String::SharedPtr msg) {

	RCLCPP_INFO(this->get_logger(), "topic receive: [%s]", msg->data.c_str());

    if (!connected_) {
        RCLCPP_WARN(this->get_logger(), "server not connected - canceal");
        return;
    }


    // "detect" or "depart" topic msg ->   [WAFFLE]detect [WAFFLE]depart  ->>tcp server
    if (msg->data == "detect" || msg->data == "depart") {
      std::string wire_msg = "[ALLMSG]" + msg->data + "\n";
      
      std::lock_guard<std::mutex> lock(socket_mutex_);
      if (sock_fd_ >= 0) {
        ::write(sock_fd_, wire_msg.c_str(), wire_msg.size());
        RCLCPP_INFO(this->get_logger(), "server report success: %s", wire_msg.c_str());
      }
    }
  }

  void reconnectIfNeeded() { if (!connected_) connectToServer(); }

  void connectToServer() {
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

	//server identify :[HSJ_WF:PASSWD] 
	std::string auth_msg = "[" + client_id_ + ":" + password_ + "]";
    ::write(sock_fd_, auth_msg.c_str(), auth_msg.size());


    connected_ = true;
    RCLCPP_INFO(this->get_logger(), "TCP server (%s:%d) connect success!", server_ip_.c_str(), server_port_);
	RCLCPP_INFO(this->get_logger(), "server connect & try identify : %s", auth_msg.c_str());

    if (!recv_thread_.joinable()) {
      stop_recv_ = false;
      recv_thread_ = std::thread(&WaffleBridgeClient::recvLoop, this);
    }
  }

  void recvLoop() {
    char buffer[256];
    while (!stop_recv_ && rclcpp::ok()) {
      int local_sock = -1;
      { std::lock_guard<std::mutex> lock(socket_mutex_); local_sock = sock_fd_; }

      if (!connected_ || local_sock < 0) { std::this_thread::sleep_for(200ms); continue; }

      std::memset(buffer, 0, sizeof(buffer));
      ssize_t len = ::read(local_sock, buffer, sizeof(buffer) - 1);
      
      if (len <= 0) {
        RCLCPP_WARN(this->get_logger(), "server disconnected!");
        connected_ = false;
        closeSocket();
        continue;
      }

      //data parsing
      std::string received_raw(buffer);
      received_raw.erase(0, received_raw.find_first_not_of(" \n\r\t"));
      received_raw.erase(received_raw.find_last_not_of(" \n\r\t") + 1);

      if (!received_raw.empty()) {
        auto msg = std_msgs::msg::String();

        //  "tts1"server message ->  "start" 
        if (received_raw == "tts1") {
          msg.data = "start";
          RCLCPP_INFO(this->get_logger(), "server 'tts1' -> 'start' pub");
        }
		else if (received_raw == "tts2"){
			msg.data = "return";
			RCLCPP_INFO(this->get_logger(), "server 'tts2' -> 'return' pub");
		}
		else if(received_raw == "tts3"){
			msg.data = "force_return";
			RCLCPP_INFO(this->get_logger(), "server 'tts2' -> 'force_return' pub");
        }
		//else {
		//	msg.data = received_raw;
		//}

		if(!msg.data.empty()){
        	wf_command_pub_->publish(msg);
			RCLCPP_INFO(this->get_logger(), "cmd published!! : %s", msg.data.c_str());
		}
      }
    }
  }

  void closeSocket() { std::lock_guard<std::mutex> lock(socket_mutex_); closeSocketUnlocked(); }
  void closeSocketUnlocked() { if (sock_fd_ >= 0) { ::close(sock_fd_); sock_fd_ = -1; } }

  std::string server_ip_, client_id_, password_;
  int server_port_, reconnect_period_ms_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr wf_command_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr wf_command_sub_;
  rclcpp::TimerBase::SharedPtr reconnect_timer_;

  std::thread recv_thread_;
  std::mutex socket_mutex_;
  int sock_fd_;
  std::atomic<bool> connected_, stop_recv_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<WaffleBridgeClient>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
