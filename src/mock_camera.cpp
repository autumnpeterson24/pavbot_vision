/*
  Mock Camera Node ===============================
  Author: Autumn Peterson (IGVC 2026 PAVbot Team)
 
  Description:
  This ROS2 node simulates a camera by looping a video file at a fixed FPS,
  uses OpenCV VideoCapture to read frames and publishes them as ROS
  sensor_msgs/Image messages via image_transport. This allows testing of
  downstream vision nodes (like lane detection) without needing physical hardware.

  Based on standard ROS2 Humble and OpenCV APIs.
  ==================================================
 */

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>

class MockCameraNode : public rclcpp::Node {
public:
  MockCameraNode() : Node("mock_camera") {
    declare_parameter<std::string>("video_path", "");
    declare_parameter<double>("fps", 30.0);
    declare_parameter<bool>("loop", true);
    declare_parameter<std::string>("frame_id", "camera_optical_frame");

    get_parameter("video_path", video_path_);
    get_parameter("fps", fps_);
    get_parameter("loop", loop_);
    get_parameter("frame_id", frame_id_);

    if (video_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "Parameter 'video_path' is empty.");
      throw std::runtime_error("video_path required");
    }

    cap_.open(video_path_);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Failed to open video: %s", video_path_.c_str());
      throw std::runtime_error("cannot open video");
    }

    img_pub_ = image_transport::create_publisher(this, "/camera/image_raw");
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/camera/camera_info", 10);

    width_  = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);

    cam_info_.header.frame_id = frame_id_;
    cam_info_.width = width_;
    cam_info_.height = height_;
    cam_info_.distortion_model = "plumb_bob";
    cam_info_.d = {0,0,0,0,0};
    cam_info_.k = {1,0,width_/2.0, 0,1,height_/2.0, 0,0,1};
    cam_info_.r = {1,0,0,0,1,0,0,0,1};
    cam_info_.p = {1,0,width_/2.0,0,0,1,height_/2.0,0,0,0,1,0};

    using namespace std::chrono_literals;
    auto period = std::chrono::milliseconds((int)(1000.0 / std::max(1.0, fps_)));
    timer_ = create_wall_timer(period, std::bind(&MockCameraNode::tick, this));

    RCLCPP_INFO(get_logger(), "MockCamera streaming %dx%d @ %.1f FPS from %s",
                width_, height_, fps_, video_path_.c_str());
  }

private:
  void tick() {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      if (loop_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(frame) || frame.empty()) return;
      } else {
        RCLCPP_INFO(get_logger(), "Video ended.");
        rclcpp::shutdown();
        return;
      }
    }

    auto stamp = now();
    auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    img_msg->header.stamp = stamp;
    img_msg->header.frame_id = frame_id_;
    img_pub_.publish(img_msg);

    cam_info_.header.stamp = stamp;
    info_pub_->publish(cam_info_);
  }

  std::string video_path_, frame_id_;
  double fps_;
  bool loop_;
  int width_{0}, height_{0};
  cv::VideoCapture cap_;
  sensor_msgs::msg::CameraInfo cam_info_;
  image_transport::Publisher img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockCameraNode>());
  rclcpp::shutdown();
  return 0;
}
