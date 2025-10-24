/*
Camera Publisher Node for Lane Detection using the ELP Sony IMX323
*/

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class CamPub : public rclcpp::Node {
public:
  CamPub() : Node("cam_pub") {
    declare_parameter<std::string>("device", "/dev/video2"); // THIS IS SUBJECT TO CHANGE BASED ON USB PORTS
    declare_parameter<int>("width", 1280);
    declare_parameter<int>("height", 720);
    declare_parameter<double>("fps", 30.0);
    declare_parameter<std::string>("fourcc", "MJPG");   // try YUYV if needed
    declare_parameter<std::string>("frame_id", "camera_optical_frame");

    get_parameter("device", device_);
    get_parameter("width", width_);
    get_parameter("height", height_);
    get_parameter("fps", fps_);
    get_parameter("fourcc", fourcc_);
    get_parameter("frame_id", frame_id_);

    // Open device via V4L2 backend
    cap_.open(device_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Failed to open device %s", device_.c_str());
      throw std::runtime_error("camera open failed");
    }

    // Set desired format
    if (!fourcc_.empty() && fourcc_.size() >= 4) {
      int four = cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]);
      cap_.set(cv::CAP_PROP_FOURCC, four);
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FPS,          fps_);

    // Warm-up: grab one frame
    cv::Mat test;
    if (!cap_.read(test) || test.empty()) {
      RCLCPP_FATAL(get_logger(), "Failed to grab initial frame from %s", device_.c_str());
      throw std::runtime_error("no frames");
    }

    img_pub_  = image_transport::create_publisher(this, "/camera/image_raw");
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/camera/camera_info", 10);

    cam_info_.header.frame_id = frame_id_;
    cam_info_.width  = static_cast<unsigned int>(width_);
    cam_info_.height = static_cast<unsigned int>(height_);
    cam_info_.distortion_model = "plumb_bob";
    cam_info_.d = {0,0,0,0,0};
    cam_info_.k = {1,0,width_/2.0, 0,1,height_/2.0, 0,0,1};
    cam_info_.r = {1,0,0,0,1,0,0,0,1};
    cam_info_.p = {1,0,width_/2.0,0, 0,1,height_/2.0,0, 0,0,1,0};

    using namespace std::chrono_literals;
    auto period = std::chrono::milliseconds((int)(1000.0 / std::max(1.0, fps_)));
    timer_ = create_wall_timer(period, std::bind(&CamPub::tick, this));

    RCLCPP_INFO(get_logger(), "cam_pub streaming %s %dx%d @ %.1f (%s)",
                device_.c_str(), width_, height_, fps_, fourcc_.c_str());
  }

private:
  void tick() {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Empty frame, retrying...");
      return;
    }

    // Normalize to 3-channel BGR for cv_bridge
    if (frame.channels() == 1) {
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
      cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    std_msgs::msg::Header hdr;
    hdr.stamp = now();
    hdr.frame_id = frame_id_;

    img_pub_.publish(cv_bridge::CvImage(hdr, "bgr8", frame).toImageMsg());
    cam_info_.header.stamp = hdr.stamp;
    info_pub_->publish(cam_info_);
  }

  // Parameters
  std::string device_, fourcc_, frame_id_;
  int width_, height_;
  double fps_;

  // ROS
  image_transport::Publisher img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Capture
  cv::VideoCapture cap_;
  sensor_msgs::msg::CameraInfo cam_info_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CamPub>());
  rclcpp::shutdown();
  return 0;
}
