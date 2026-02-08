// pothole_points_bridge.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>


class PotholePointsBridge : public rclcpp::Node {
public:
  PotholePointsBridge() : Node("pothole_points_bridge") {
    declare_parameter<std::string>("in_topic",  "/potholes/poses");
    declare_parameter<std::string>("out_topic", "/potholes/points");
    declare_parameter<std::string>("frame_id",  "base_link");

    in_topic_  = get_parameter("in_topic").as_string();
    out_topic_ = get_parameter("out_topic").as_string();
    frame_id_  = get_parameter("frame_id").as_string();

    sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      in_topic_, 10,
      std::bind(&PotholePointsBridge::cb, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, 10);

    RCLCPP_INFO(get_logger(), "Bridge %s -> %s", in_topic_.c_str(), out_topic_.c_str());
  }

private:
  void cb(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = msg->header.stamp;
    cloud.header.frame_id = msg->header.frame_id.empty() ? frame_id_ : msg->header.frame_id;

    cloud.height = 1;
    cloud.width  = static_cast<uint32_t>(msg->poses.size());
    cloud.is_dense = true;

    sensor_msgs::PointCloud2Modifier mod(cloud);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(cloud.width);

    sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iy(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iz(cloud, "z");

    for (const auto& p : msg->poses) {
      *ix = static_cast<float>(p.position.x);
      *iy = static_cast<float>(p.position.y);
      *iz = 0.0f;
      ++ix; ++iy; ++iz;
    }

    pub_->publish(cloud);
  }

  std::string in_topic_, out_topic_, frame_id_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PotholePointsBridge>());
  rclcpp::shutdown();
  return 0;
}
