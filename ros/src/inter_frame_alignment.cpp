#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>

#include "./tictoc.hpp"
#include "slam/loop_closure.h"
#include "slam/utils.hpp"

using namespace kiss_matcher;

class InterFrameAligner : public rclcpp::Node {
 public:
  explicit InterFrameAligner(const rclcpp::NodeOptions &options)
      : rclcpp::Node("inter_frame_aligner", options) {
    double frame_update_hz;

    LoopClosureConfig lc_config;

    auto &gc = lc_config.gicp_config_;
    auto &mc = lc_config.matcher_config_;

    source_frame_   = declare_parameter<std::string>("source_frame", "");
    target_frame_   = declare_parameter<std::string>("target_frame", "");
    frame_update_hz = declare_parameter<double>("frame_update_hz", 0.2);

    lc_config.voxel_res_ = declare_parameter<double>("voxel_resolution", 1.0);
    lc_config.verbose_   = declare_parameter<bool>("loop.verbose", false);

    gc.num_threads_               = declare_parameter<int>("local_reg.num_threads", 8);
    gc.correspondence_randomness_ = declare_parameter<int>("local_reg.correspondences_number", 20);
    gc.max_num_iter_              = declare_parameter<int>("local_reg.max_num_iter", 32);
    gc.scale_factor_for_corr_dist_ =
        declare_parameter<double>("local_reg.scale_factor_for_corr_dist", 5.0);
    gc.overlap_threshold_ = declare_parameter<double>("local_reg.overlap_threshold", 90.0);

    lc_config.enable_global_registration_ = declare_parameter<bool>("global_reg.enable", false);
    lc_config.num_inliers_threshold_ =
        declare_parameter<int>("global_reg.num_inliers_threshold", 100);

    rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    reg_module_ = std::make_shared<LoopClosure>(lc_config, this->get_logger());

    debug_src_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inter_frame/src", 10);
    debug_tgt_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("inter_frame/tgt", 10);
    debug_coarse_aligned_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("inter_frame/coarse_alignment", 10);
    debug_fine_aligned_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("inter_frame/fine_alignment", 10);
    debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lc/debug_cloud", 10);

    inter_alignment_timer_ =
        this->create_wall_timer(std::chrono::duration<double>(1.0 / frame_update_hz),
                                std::bind(&InterFrameAligner::performAlignment, this));

    // 20 Hz is enough as long as it's faster than the full registration process.
    cloud_vis_timer_ =
        this->create_wall_timer(std::chrono::duration<double>(1.0 / 20.0),
                                std::bind(&InterFrameAligner::visualizeClouds, this));

    sub_source_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "source", qos, std::bind(&InterFrameAligner::callbackSource, this, std::placeholders::_1));
    sub_target_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "target", qos, std::bind(&InterFrameAligner::callbackTarget, this, std::placeholders::_1));

    source_cloud_.reset(new pcl::PointCloud<PointType>());
    target_cloud_.reset(new pcl::PointCloud<PointType>());
  }

  void callbackSource(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    pcl::fromROSMsg(*msg, *source_cloud_);
    is_source_updated_ = true;
  }

  void callbackTarget(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    pcl::fromROSMsg(*msg, *target_cloud_);
    is_target_updated_ = true;
  }

  void performAlignment() {
    if (!is_source_updated_ || !is_target_updated_) return;

    const auto &estimate = reg_module_->coarseToFineAlignment(*source_cloud_, *target_cloud_);
    const double eps     = 1e-6;
    if ((estimate.pose_ - Eigen::Matrix4d::Identity()).norm() < eps) {
      RCLCPP_INFO(this->get_logger(), "Pose is approximately identity.");
    }

    target_T_source_ = estimate.pose_;

    Eigen::Matrix3d rot   = target_T_source_.block<3, 3>(0, 0);
    Eigen::Vector3d trans = target_T_source_.block<3, 1>(0, 3);

    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp    = this->get_clock()->now();
    transform_msg.header.frame_id = target_frame_;
    transform_msg.child_frame_id  = source_frame_;

    transform_msg.transform.translation.x = trans.x();
    transform_msg.transform.translation.y = trans.y();
    transform_msg.transform.translation.z = trans.z();

    Eigen::Quaterniond q(rot);
    transform_msg.transform.rotation.x = q.x();
    transform_msg.transform.rotation.y = q.y();
    transform_msg.transform.rotation.z = q.z();
    transform_msg.transform.rotation.w = q.w();

    tf_broadcaster_->sendTransform(transform_msg);

    is_source_updated_ = false;
    is_target_updated_ = false;

    need_lc_cloud_vis_update_ = true;
  }

  void visualizeClouds() {
    if (!need_lc_cloud_vis_update_) {
      return;
    }
    debug_src_pub_->publish(toROSMsg(reg_module_->getSourceCloud(), target_frame_));
    debug_tgt_pub_->publish(toROSMsg(reg_module_->getTargetCloud(), target_frame_));
    debug_fine_aligned_pub_->publish(toROSMsg(reg_module_->getFinalAlignedCloud(), target_frame_));
    debug_coarse_aligned_pub_->publish(
        toROSMsg(reg_module_->getCoarseAlignedCloud(), target_frame_));
    debug_cloud_pub_->publish(toROSMsg(reg_module_->getDebugCloud(), target_frame_));
    need_lc_cloud_vis_update_ = false;
  }

 private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_source_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_target_;

  std::string source_frame_;
  std::string target_frame_;

  pcl::PointCloud<PointType>::Ptr source_cloud_;
  pcl::PointCloud<PointType>::Ptr target_cloud_;

  bool is_source_updated_ = false;
  bool is_target_updated_ = false;

  bool need_lc_cloud_vis_update_ = false;

  std::shared_ptr<kiss_matcher::LoopClosure> reg_module_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  kiss_matcher::TicToc timer_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_src_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_tgt_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_coarse_aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_fine_aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;

  rclcpp::TimerBase::SharedPtr inter_alignment_timer_;
  rclcpp::TimerBase::SharedPtr cloud_vis_timer_;

  Eigen::Matrix4d target_T_source_ = Eigen::Matrix4d::Identity();
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;

  auto node = std::make_shared<InterFrameAligner>(options);

  // To allow timer callbacks to run concurrently using multiple threads
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
