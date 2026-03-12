#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>

// 네 srv 패키지/이름에 맞게 include 경로가 달라짐
#include "omx_pick_place/srv/pose_to_joint_execute.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
using PoseToJointExecute = omx_pick_place::srv::PoseToJointExecute;
using FJT = control_msgs::action::FollowJointTrajectory;
using GoalHandleFJT = rclcpp_action::ClientGoalHandle<FJT>;
class PoseToJointExecuteServer : public rclcpp::Node
{
public:
  PoseToJointExecuteServer()
  : Node("pose_to_joint_execute_server"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // MoveGroupInterface는 shared_from_this() 때문에 타이머 init 권장
    init_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&PoseToJointExecuteServer::init_once, this));
    // FollowJointTrajectory action 이름 (moveit_controllers.yaml 기준 arm_controller/follow_joint_trajectory)
    fjt_action_name_ = this->declare_parameter<std::string>(
      "fjt_action_name", "/arm_controller/follow_joint_trajectory");

    fjt_client_ = rclcpp_action::create_client<FJT>(this, fjt_action_name_);

    RCLCPP_INFO(get_logger(), "Direct FJT action: %s", fjt_action_name_.c_str());
  }
  ~PoseToJointExecuteServer() override
  {
    if (mg_exec_ && mg_node_) {
      mg_exec_->cancel();
      mg_exec_->remove_node(mg_node_);
    }
    if (mg_spin_thread_.joinable()) {
      mg_spin_thread_.join();
    }
  }
private:
  void init_once()
  {
    init_timer_->cancel();

    group_name_ = this->declare_parameter<std::string>("move_group", "arm");

    // ★ MoveIt 전용 노드 생성 (중요)
    mg_node_ = std::make_shared<rclcpp::Node>("pose_to_joint_mg_node");

    // ★ 전용 executor + spin thread
    mg_exec_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    mg_exec_->add_node(mg_node_);
    mg_spin_thread_ = std::thread([this]() { mg_exec_->spin(); });

    // ★ MoveGroupInterface는 mg_node_로 만든다
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      mg_node_, group_name_);

    // action client는 서비스 노드(this)에 둬도 되고 mg_node_에 둬도 되는데
    // 보통 this에 둬도 OK. (아래는 기존대로)
    if (this->has_parameter("fjt_action_name")) {
      fjt_action_name_ = this->get_parameter("fjt_action_name").as_string();
    } else {
      fjt_action_name_ = this->declare_parameter<std::string>(
        "fjt_action_name", "/arm_controller/follow_joint_trajectory");
    }
    action_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    fjt_client_ = rclcpp_action::create_client<FJT>(
      this->get_node_base_interface(),
      this->get_node_graph_interface(),
      this->get_node_logging_interface(),
      this->get_node_waitables_interface(),
      fjt_action_name_,
      action_cb_group_);
    move_group_->setPlanningTime(5.0);
    move_group_->setMaxVelocityScalingFactor(0.3);
    move_group_->setMaxAccelerationScalingFactor(0.3);

    RCLCPP_INFO(get_logger(), "MoveGroup ready. group=%s planning_frame=%s ee=%s",
                move_group_->getName().c_str(),
                move_group_->getPlanningFrame().c_str(),
                move_group_->getEndEffectorLink().c_str());

    service_ = this->create_service<PoseToJointExecute>(
      "pose_to_joint_execute",
      std::bind(&PoseToJointExecuteServer::handle_request, this,
                std::placeholders::_1, std::placeholders::_2));
    js_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 50,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg){
        std::lock_guard<std::mutex> lk(js_mutex_);
        last_js_ = *msg;
      });
  }

  void handle_request(
    const std::shared_ptr<PoseToJointExecute::Request> req,
    std::shared_ptr<PoseToJointExecute::Response> res)
  {
    res->success = false;
    res->message.clear();
    res->joint_state = sensor_msgs::msg::JointState();

    if (!move_group_) {
      res->message = "MoveGroup not initialized yet";
      return;
    }

    // 1) target pose를 planning frame으로 변환 (요청 pose의 frame_id가 다를 수 있음)
    geometry_msgs::msg::PoseStamped target_plan;
    if (!transformPoseToPlanningFrame(req->target_pose, target_plan)) {
      res->message = "TF transform to planning frame failed";
      return;
    }

    // 2) IK로 joint values 만들기
    std::vector<std::string> joint_names;
    std::vector<double> joint_values;
    double ik_timeout = req->ik_timeout_sec > 0.0 ? req->ik_timeout_sec : 0.2;

    if (!computeIKJointValues(target_plan, joint_names, joint_values, ik_timeout)) {
      res->message = "IK failed. Try: increase ik_timeout_sec / relax orientation / check reachability";
      return;
    }
    //joint5=0
    for (size_t i = 0; i < joint_names.size(); ++i) {
      if (joint_names[i] == "joint5") {
        joint_values[i] = 0.0;
        break;
      }
    }
    // 3) response에 joint_state 채우기
    res->joint_state.header.stamp = now();
    res->joint_state.name = joint_names;
    res->joint_state.position = joint_values;
    // 4) execute_mode에 따라 동작
    // 0: IK only
    if (req->execute_mode == 0) {
      res->success = true;
      res->message = "IK ok (mode=IK_ONLY)";
      return;
    }

    // 1: MoveIt plan+execute
    if (req->execute_mode == 1) {
      std::string msg;
      if (!planAndExecuteJointTarget(joint_values, msg)) {
        res->success = false;
        res->message = msg.empty() ? "MoveIt plan/execute failed" : msg;
        return;
      }
      res->success = true;
      res->message = "IK ok, MoveIt executed (mode=MOVEIT_PLAN_EXECUTE)";
      return;
    }

    // 2: Direct controller execution (no planning)
    if (req->execute_mode == 2) {
      double duration = (req->move_duration_sec > 0.0) ? req->move_duration_sec : 2.0;

      std::string msg;
      if (!directExecuteByFJT(joint_names, joint_values, duration, msg)) {
        res->success = false;
        res->message = msg.empty() ? "Direct controller execution failed" : msg;
        return;
      }

      res->success = true;
      res->message = "IK ok, direct executed (mode=DIRECT_CONTROLLER)";
      return;
    }

    res->success = false;
    res->message = "Invalid execute_mode (use 0,1,2)";
    return;

  }

  bool transformPoseToPlanningFrame(
    const geometry_msgs::msg::PoseStamped& in_pose,
    geometry_msgs::msg::PoseStamped& out_pose)
  {
    const std::string planning_frame = move_group_->getPlanningFrame();

    if (in_pose.header.frame_id.empty()) {
      RCLCPP_ERROR(get_logger(), "Request pose frame_id is empty");
      return false;
    }

    if (in_pose.header.frame_id == planning_frame) {
      out_pose = in_pose;
      return true;
    }

    try {
      const auto tf = tf_buffer_.lookupTransform(
        planning_frame, in_pose.header.frame_id, tf2::TimePointZero);

      tf2::doTransform(in_pose, out_pose, tf);
      out_pose.header.frame_id = planning_frame;
      out_pose.header.stamp = now();
      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "TF failed: %s", e.what());
      return false;
    }
  }

  bool computeIKJointValues(
    const geometry_msgs::msg::PoseStamped& target_in_planning_frame,
    std::vector<std::string>& out_joint_names,
    std::vector<double>& out_joint_values,
    double ik_timeout_sec)
  {
    const auto robot_model = move_group_->getRobotModel();
    const auto* jmg = robot_model->getJointModelGroup(move_group_->getName());
    if (!jmg) {
      RCLCPP_ERROR(get_logger(), "JointModelGroup not found: %s", move_group_->getName().c_str());
      return false;
    }
    moveit::core::RobotState state(robot_model);
    state.setToDefaultValues();

    // ★ 최근 joint_states를 seed로 반영
    sensor_msgs::msg::JointState js_copy;
    {
      std::lock_guard<std::mutex> lk(js_mutex_);
      js_copy = last_js_;
    }

    if (!js_copy.name.empty() && js_copy.name.size() == js_copy.position.size()) {
      // MoveIt의 variable order는 URDF/MoveIt에 의해 정해지므로
      // 이름 매칭으로 값을 넣어준다
      for (size_t i = 0; i < js_copy.name.size(); ++i) {
        const std::string& jn = js_copy.name[i];
        double pos = js_copy.position[i];
	
	try {
    	     (void)robot_model->getVariableIndex(jn);  // 없으면 예외 throw :contentReference[oaicite:2]{index=2}
             state.setVariablePosition(jn, pos);       // 존재하면 세팅
        } catch (const std::exception&) {
        // 모델에 없는 joint/variable 이름이면 무시
          continue;
        }
       }
       state.update();
    } else {
      RCLCPP_WARN(get_logger(), "No joint_states received yet -> using default seed.");
    }

    // IK
    bool ok = state.setFromIK(jmg, target_in_planning_frame.pose, ik_timeout_sec);
    if (!ok) return false;

    out_joint_names = jmg->getVariableNames();
    state.copyJointGroupPositions(jmg, out_joint_values);
    return true;
  }

  bool planAndExecuteJointTarget(const std::vector<double>& joint_values, std::string& out_msg)
  {
    move_group_->setJointValueTarget(joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto plan_code = move_group_->plan(plan);
    if (plan_code != moveit::core::MoveItErrorCode::SUCCESS) {
      out_msg = "Planning failed (MoveItErrorCode=" + std::to_string(plan_code.val) + ")";
      return false;
    }

    auto exec_code = move_group_->execute(plan);
    if (exec_code != moveit::core::MoveItErrorCode::SUCCESS) {
      out_msg = "Execute failed (MoveItErrorCode=" + std::to_string(exec_code.val) + ")";
      return false;
    }

    return true;
  }
  bool directExecuteByFJT(const std::vector<std::string>& joint_names,
                          const std::vector<double>& joint_positions,
                          double duration_sec,
                          std::string& out_msg)
  {
    if (!fjt_client_) {
      out_msg = "FJT action client not initialized";
      return false;
    }

    if (joint_names.empty() || joint_positions.empty() || joint_names.size() != joint_positions.size()) {
      out_msg = "Invalid joint_names/positions";
      return false;
    }

    if (!fjt_client_->wait_for_action_server(std::chrono::seconds(2))) {
      out_msg = "FollowJointTrajectory action server not available: " + fjt_action_name_;
      return false;
    }

    FJT::Goal goal;
    goal.trajectory.joint_names = joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint p;
    p.positions = joint_positions;
    p.time_from_start = rclcpp::Duration::from_seconds(duration_sec);
    goal.trajectory.points.push_back(p);

    // 동기적으로 결과 기다리기(서비스 요청-응답이라 깔끔)
    auto goal_future = fjt_client_->async_send_goal(goal);
    if (goal_future.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
      out_msg = "Failed to send FJT goal (timeout)";
      return false;
    }

    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      out_msg = "FJT goal was rejected by server";
      return false;
    }

    auto result_future = fjt_client_->async_get_result(goal_handle);
    // duration + 여유
    auto wait_time = std::chrono::milliseconds(static_cast<int>((duration_sec + 2.0) * 1000.0));
    if (result_future.wait_for(wait_time) != std::future_status::ready) {
    out_msg = "FJT result timeout";
    return false;
    }

    auto wrapped = result_future.get();
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
      out_msg = "FJT failed (ResultCode=" + std::to_string(static_cast<int>(wrapped.code)) + ")";
      return false;
    }

    return true;
  }
private:
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::Service<PoseToJointExecute>::SharedPtr service_;

  rclcpp::Node::SharedPtr mg_node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> mg_exec_;
  std::thread mg_spin_thread_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  sensor_msgs::msg::JointState last_js_;
  std::mutex js_mutex_;
  
  std::string group_name_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  rclcpp::CallbackGroup::SharedPtr action_cb_group_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp_action::Client<FJT>::SharedPtr fjt_client_;
  std::string fjt_action_name_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PoseToJointExecuteServer>();

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
