#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>
#include <casadi/casadi.hpp>

#include <algorithm>
#include <vector>
#include <iostream>

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;
using namespace casadi;

SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
  // パラメーターの初期化
  wheel_base_(declare_parameter<float>("wheel_base", 2.14)),
  lookahead_gain_(declare_parameter<float>("lookahead_gain", 1.0)),
  lookahead_min_distance_(declare_parameter<float>("lookahead_min_distance", 1.0)),
  speed_proportional_gain_(declare_parameter<float>("speed_proportional_gain", 1.0)),
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<float>("external_target_vel", 0.0))
{
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);

  sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", 1, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
  sub_trajectory_ = create_subscription<Trajectory>(
    "input/trajectory", 1, [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });

  using namespace std::literals::chrono_literals;
  timer_ =
    rclcpp::create_timer(this, get_clock(), 30ms, std::bind(&SimplePurePursuit::onTimer, this));

  // MPCの初期化
  // initialize_mpc();
  isInit = false;
}

/*
void SimplePurePursuit::initialize_mpc() {
  N_ = 10;  // 予測ホライズン
  dt_ = 0.03;  // タイムステップをタイマの設定と一致させる

  // 状態変数と制御変数の定義
  MX x = MX::sym("x", 4);  // 例：状態変数 [x, y, theta, v]
  MX u = MX::sym("u", 2);  // 例：制御変数 [a, delta]

  // 動的モデルの定義
  MX x_next = MX::vertcat({
    x(3) * cos(x(2)),
    x(3) * sin(x(2)),
    u(1),  // 角速度
    u(0)   // 加速度
  });

  // MPC最適化問題のセットアップ
  MX X = MX::sym("X", 4, N_ + 1);
  MX U = MX::sym("U", 2, N_);
  MX obj = 0;
  std::vector<MX> g;

  DM Q = DM::diagcat({1, 1, 0.1, 0.1});
  DM R = DM::diagcat({0.1, 0.1});
  DM x_ref_ = DM({10, 10, 0, 5});  // 例：目標状態（速度を5に設定）

  for (int k = 0; k < N_; ++k) {
    // x_next の計算に現在の状態変数 X を適用する
    MX x_next_k = X(Slice(), k) + dt_ * MX::vertcat({
      X(3, k) * cos(X(2, k)),
      X(3, k) * sin(X(2, k)),
      U(1, k),  // 角速度
      U(0, k)   // 加速度
    });
    g.push_back(X(Slice(), k + 1) - x_next_k);
    obj += mtimes((X(Slice(), k) - x_ref_).T(), mtimes(Q, X(Slice(), k) - x_ref_)) +
           mtimes(U(Slice(), k).T(), mtimes(R, U(Slice(), k)));
  }

  MXDict nlp = {{"x", MX::vertcat({reshape(X, -1, 1), reshape(U, -1, 1)})},
                {"f", obj},
                {"g", MX::vertcat(g)}};

  solver_ = nlpsol("solver", "ipopt", nlp);
}
*/

void SimplePurePursuit::initialize_mpc(const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> &trajectory, size_t closest_traj_point_idx) {
  N_ = 10;  // 予測ホライズン
  dt_ = 0.03;  // タイムステップをタイマの設定と一致させる

  // 状態変数と制御変数の定義
  MX x = MX::sym("x", 4);  // 例：状態変数 [x, y, theta, v]
  MX u = MX::sym("u", 2);  // 例：制御変数 [a, delta]

  // 動的モデルの定義
  MX x_next = MX::vertcat({
    x(3) * cos(x(2)),
    x(3) * sin(x(2)),
    u(1),  // 角速度
    u(0)   // 加速度
  });

  // closest_traj_point_idx から初回の目標位置と速度を設定
  std::vector<double> x_ref(4);
  x_ref[0] = trajectory[closest_traj_point_idx].pose.position.x;
  x_ref[1] = trajectory[closest_traj_point_idx].pose.position.y;
  x_ref[2] = tf2::getYaw(trajectory[closest_traj_point_idx].pose.orientation);
  x_ref[3] = trajectory[closest_traj_point_idx].longitudinal_velocity_mps;

  x_ref_ = DM(x_ref);

  // MPC最適化問題のセットアップ
  MX X = MX::sym("X", 4, N_ + 1);
  MX U = MX::sym("U", 2, N_);
  MX obj = 0;
  std::vector<MX> g;

  DM Q = DM::diagcat({10, 10, 1, 1});  // 重み付け行列 Q を調整
  DM R = DM::diagcat({1, 1});  // 重み付け行列 R を調整

  for (int k = 0; k < N_; ++k) {
    // x_next の計算に現在の状態変数 X を適用する
    MX x_next_k = X(Slice(), k) + dt_ * MX::vertcat({
      X(3, k) * cos(X(2, k)),
      X(3, k) * sin(X(2, k)),
      U(1, k),  // 角速度
      U(0, k)   // 加速度
    });
    g.push_back(X(Slice(), k + 1) - x_next_k);
    obj += mtimes((X(Slice(), k) - x_ref_).T(), mtimes(Q, X(Slice(), k) - x_ref_)) +
           mtimes(U(Slice(), k).T(), mtimes(R, U(Slice(), k)));
  }

  MXDict nlp = {{"x", MX::vertcat({reshape(X, -1, 1), reshape(U, -1, 1)})},
                {"f", obj},
                {"g", MX::vertcat(g)}};

  solver_ = nlpsol("solver", "ipopt", nlp);
}

std::vector<double> SimplePurePursuit::compute_control(const std::vector<double> &state_estimate) {
  size_t closest_traj_point_idx = findNearestIndex(trajectory_->points, odometry_->pose.pose.position);

  if (closest_traj_point_idx == trajectory_->points.size() - 1 || trajectory_->points.size() <= 5) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "reached to the goal");
    return {0.0, -10.0, 0.0}; // 目標に到達またはポイントが少ない場合、停止
  }

  if (isInit == false) {
    std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> trajectory_points(trajectory_->points.begin(), trajectory_->points.end());
    initialize_mpc(trajectory_points, closest_traj_point_idx);
    isInit = true;
  }

  std::vector<double> x_ref(4);
  x_ref[0] = trajectory_->points[closest_traj_point_idx].pose.position.x;
  x_ref[1] = trajectory_->points[closest_traj_point_idx].pose.position.y;
  x_ref[2] = tf2::getYaw(trajectory_->points[closest_traj_point_idx].pose.orientation);
  x_ref[3] = trajectory_->points[closest_traj_point_idx].longitudinal_velocity_mps;

  x_ref_ = DM(x_ref);

  // 初期推定値を設定（状態変数と制御変数の両方を含む）
  std::vector<double> x0_vec(4 * (N_ + 1) + 2 * N_, 0.0);
  // 状態変数の初期値を設定
  std::copy(state_estimate.begin(), state_estimate.end(), x0_vec.begin());

  DM x0 = DM(x0_vec);

  DMDict arg = {{"lbx", DM::ones(4 * (N_ + 1) + 2 * N_) * -inf},
                {"ubx", DM::ones(4 * (N_ + 1) + 2 * N_) * inf},
                {"lbg", DM::zeros(4 * N_)},
                {"ubg", DM::zeros(4 * N_)},
                {"x0", x0}};

  // デバッグ情報の表示
  std::cout << "Solver arguments:" << std::endl;
  std::cout << "lbx: " << arg["lbx"] << std::endl;
  std::cout << "ubx: " << arg["ubx"] << std::endl;
  std::cout << "lbg: " << arg["lbg"] << std::endl;
  std::cout << "ubg: " << arg["ubg"] << std::endl;
  std::cout << "x0: " << arg["x0"] << std::endl;

  DMDict res;
  try {
    res = solver_(arg);
  } catch (const std::exception &e) {
    std::cerr << "Solver failed: " << e.what() << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "lbx: " << arg["lbx"] << std::endl;
    std::cerr << "ubx: " << arg["ubx"] << std::endl;
    std::cerr << "lbg: " << arg["lbg"] << std::endl;
    std::cerr << "ubg: " << arg["ubg"] << std::endl;
    std::cerr << "x0: " << arg["x0"] << std::endl;
    return {0.0, 0.05, 0.0};  // エラーが発生した場合、制御信号をデフォルト値に設定
  }

  DM u_opt = res["x"](Slice(4 * (N_ + 1), 4 * (N_ + 1) + 2));
  std::vector<double> control_signal(static_cast<size_t>(u_opt.size1()));
  for (size_t i = 0; i < control_signal.size(); ++i) {
    control_signal[i] = static_cast<double>(u_opt(i));
  }

  // デバッグ情報の表示
  std::cout << "State Estimate: ";
  for (const auto &v : state_estimate) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  std::cout << "Control Signal: ";
  for (const auto &v : control_signal) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  std::cout << "Closest Trajectory Point Index: " << closest_traj_point_idx << std::endl;
  std::cout << "Reference State: ";
  for (const auto &v : x_ref) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  return control_signal;
}

void SimplePurePursuit::onTimer()
{
  // データのチェック
  if (!subscribeMessageAvailable()) {
    return;
  }

  // 現在の状態を取得
  std::vector<double> state_estimate = {
    odometry_->pose.pose.position.x,
    odometry_->pose.pose.position.y,
    tf2::getYaw(odometry_->pose.pose.orientation),
    odometry_->twist.twist.linear.x
  };

  // MPCを用いて制御信号を計算
  std::vector<double> control_signal = compute_control(state_estimate);

  // 制御コマンドの構築とパブリッシュ
  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());
  cmd.longitudinal.speed = control_signal[0];
  cmd.longitudinal.acceleration = control_signal[1];
  cmd.lateral.steering_tire_angle = control_signal[2];

  pub_cmd_->publish(cmd);
}

bool SimplePurePursuit::subscribeMessageAvailable()
{
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "odometry is not available");
    return false;
  }
  if (!trajectory_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "trajectory is not available");
    return false;
  }
  return true;
}

AckermannControlCommand SimplePurePursuit::zeroAckermannControlCommand(rclcpp::Time stamp)
{
  AckermannControlCommand cmd;
  cmd.stamp = stamp;
  cmd.longitudinal.stamp = stamp;
  cmd.longitudinal.speed = 0.0;
  cmd.longitudinal.acceleration = 0.0;
  cmd.lateral.stamp = stamp;
  cmd.lateral.steering_tire_angle = 0.0;
  return cmd;
}

}  // namespace simple_pure_pursuit

int main(int argc, char const * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_pure_pursuit::SimplePurePursuit>());
  rclcpp::shutdown();
  return 0;
}
