#ifndef SIMPLE_PURE_PURSUIT_HPP_
#define SIMPLE_PURE_PURSUIT_HPP_

#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <casadi/casadi.hpp>

namespace simple_pure_pursuit {

using autoware_auto_control_msgs::msg::AckermannControlCommand;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Twist;
using nav_msgs::msg::Odometry;

class SimplePurePursuit : public rclcpp::Node {
 public:
  explicit SimplePurePursuit();
  
  // サブスクリプション
  rclcpp::Subscription<Odometry>::SharedPtr sub_kinematics_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;
  
  // パブリッシャー
  rclcpp::Publisher<AckermannControlCommand>::SharedPtr pub_cmd_;
  
  // タイマー
  rclcpp::TimerBase::SharedPtr timer_;

  // サブスクリプションで更新されるデータ
  Trajectory::SharedPtr trajectory_;
  Odometry::SharedPtr odometry_;

  // パラメーター
  const double wheel_base_;
  const double lookahead_gain_;
  const double lookahead_min_distance_;
  const double speed_proportional_gain_;
  const bool use_external_target_vel_;
  const double external_target_vel_;

 private:
  void onTimer();
  bool subscribeMessageAvailable();

  // MPC固有のメンバー
  //void initialize_mpc();
  void initialize_mpc(const std::vector<TrajectoryPoint> &trajectory, size_t closest_traj_point_idx);
  std::vector<double> compute_control(const std::vector<double> &state_estimate);
  casadi::Function solver_;
  casadi::DM x_ref_;
  int N_;
  double dt_;
  bool isInit; 

  AckermannControlCommand zeroAckermannControlCommand(rclcpp::Time stamp);
};

}  // namespace simple_pure_pursuit

#endif  // SIMPLE_PURE_PURSUIT_HPP_
