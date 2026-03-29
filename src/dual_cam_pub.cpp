/*
Dual Camera Publisher Node for PAVBot
Publishes:
  /left_cam/image_raw
  /right_cam/image_raw
  /left_cam/camera_info
  /right_cam/camera_info

Designed for dual USB IMX323 cameras for lane detection + pothole detection
*/

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

class DualCamPub : public rclcpp::Node {
public:
  DualCamPub() : Node("dual_cam_pub") {
    // ---------------- General params ----------------
    declare_parameter<double>("fps", 30.0);
    declare_parameter<int>("width", 1280);
    declare_parameter<int>("height", 720);
    declare_parameter<std::string>("fourcc", "MJPG");

    // ---------------- Left camera params ----------------
    declare_parameter<std::string>("left.device", "/dev/video0");
    declare_parameter<std::string>("left.frame_id", "left_camera_link");
    declare_parameter<std::string>("left.image_topic", "/left_cam/image_raw");
    declare_parameter<std::string>("left.info_topic", "/left_cam/camera_info");

    declare_parameter<double>("left.fx", 1.0);
    declare_parameter<double>("left.fy", 1.0);
    declare_parameter<double>("left.cx", 640.0);
    declare_parameter<double>("left.cy", 360.0);
    declare_parameter<std::vector<double>>("left.d", std::vector<double>{0,0,0,0,0});

    // ---------------- Right camera params ----------------
    declare_parameter<std::string>("right.device", "/dev/video2");
    declare_parameter<std::string>("right.frame_id", "right_camera_link");
    declare_parameter<std::string>("right.image_topic", "/right_cam/image_raw");
    declare_parameter<std::string>("right.info_topic", "/right_cam/camera_info");

    declare_parameter<double>("right.fx", 1.0);
    declare_parameter<double>("right.fy", 1.0);
    declare_parameter<double>("right.cx", 640.0);
    declare_parameter<double>("right.cy", 360.0);
    declare_parameter<std::vector<double>>("right.d", std::vector<double>{0,0,0,0,0});

    // ---------------- Load general params ----------------
    get_parameter("fps", fps_);
    get_parameter("width", width_);
    get_parameter("height", height_);
    get_parameter("fourcc", fourcc_);

    // ---------------- Load left params ----------------
    get_parameter("left.device", left_.device);
    get_parameter("left.frame_id", left_.frame_id);
    get_parameter("left.image_topic", left_.image_topic);
    get_parameter("left.info_topic", left_.info_topic);

    get_parameter("left.fx", left_.fx);
    get_parameter("left.fy", left_.fy);
    get_parameter("left.cx", left_.cx);
    get_parameter("left.cy", left_.cy);
    get_parameter("left.d", left_.d);

    // ---------------- Load right params ----------------
    get_parameter("right.device", right_.device);
    get_parameter("right.frame_id", right_.frame_id);
    get_parameter("right.image_topic", right_.image_topic);
    get_parameter("right.info_topic", right_.info_topic);

    get_parameter("right.fx", right_.fx);
    get_parameter("right.fy", right_.fy);
    get_parameter("right.cx", right_.cx);
    get_parameter("right.cy", right_.cy);
    get_parameter("right.d", right_.d);

    // Open both cameras
    open_camera(left_, "left");
    open_camera(right_, "right");

    // Warm-up grab
    warmup_camera(left_, "left");
    warmup_camera(right_, "right");

    // Publishers
    left_.img_pub  = image_transport::create_publisher(this, left_.image_topic);
    left_.info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(left_.info_topic, 10);

    right_.img_pub  = image_transport::create_publisher(this, right_.image_topic);
    right_.info_pub = create_publisher<sensor_msgs::msg::CameraInfo>(right_.info_topic, 10);

    // CameraInfo setup
    init_camera_info(left_);
    init_camera_info(right_);

    using namespace std::chrono_literals;
    auto period = std::chrono::milliseconds(
      static_cast<int>(1000.0 / std::max(1.0, fps_))
    );

    timer_ = create_wall_timer(period, std::bind(&DualCamPub::tick, this));

    RCLCPP_INFO(get_logger(), "dual_cam_pub started");
    RCLCPP_INFO(get_logger(), "Left : %s -> %s",
                left_.device.c_str(), left_.image_topic.c_str());
    RCLCPP_INFO(get_logger(), "Right: %s -> %s",
                right_.device.c_str(), right_.image_topic.c_str());
    RCLCPP_INFO(get_logger(), "Streaming %dx%d @ %.1f FPS (%s)",
                width_, height_, fps_, fourcc_.c_str());
  }

private:
  struct CameraSide {
    std::string device;
    std::string frame_id;
    std::string image_topic;
    std::string info_topic;

    double fx{1.0};
    double fy{1.0};
    double cx{0.0};
    double cy{0.0};
    std::vector<double> d{0,0,0,0,0};

    image_transport::Publisher img_pub;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub;

    cv::VideoCapture cap;
    sensor_msgs::msg::CameraInfo cam_info;
  };

  void open_camera(CameraSide & cam, const std::string & name) {
    cam.cap.open(cam.device, cv::CAP_V4L2);
    if (!cam.cap.isOpened()) {
      RCLCPP_FATAL(get_logger(), "[%s] Failed to open device %s",
                   name.c_str(), cam.device.c_str());
      throw std::runtime_error("camera open failed");
    }

    if (!fourcc_.empty() && fourcc_.size() >= 4) {
      int four = cv::VideoWriter::fourcc(
        fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]
      );
      cam.cap.set(cv::CAP_PROP_FOURCC, four);
    }

    cam.cap.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    cam.cap.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cam.cap.set(cv::CAP_PROP_FPS, fps_);
    cam.cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    int actual_w = static_cast<int>(cam.cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_h = static_cast<int>(cam.cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cam.cap.get(cv::CAP_PROP_FPS);

    RCLCPP_INFO(get_logger(), "[%s] Opened %s at %dx%d @ %.2f FPS",
                name.c_str(), cam.device.c_str(), actual_w, actual_h, actual_fps);
  }

  void warmup_camera(CameraSide & cam, const std::string & name) {
    cv::Mat test;
    if (!cam.cap.read(test) || test.empty()) {
      RCLCPP_FATAL(get_logger(), "[%s] Failed to grab initial frame from %s",
                   name.c_str(), cam.device.c_str());
      throw std::runtime_error("no frames");
    }
  }

  void init_camera_info(CameraSide & cam) {
    cam.cam_info.header.frame_id = cam.frame_id;
    cam.cam_info.width = static_cast<unsigned int>(width_);
    cam.cam_info.height = static_cast<unsigned int>(height_);
    cam.cam_info.distortion_model = "plumb_bob";
    cam.cam_info.d = cam.d;

    cam.cam_info.k = {
      cam.fx, 0.0,    cam.cx,
      0.0,    cam.fy, cam.cy,
      0.0,    0.0,    1.0
    };

    cam.cam_info.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0
    };

    cam.cam_info.p = {
      cam.fx, 0.0,    cam.cx, 0.0,
      0.0,    cam.fy, cam.cy, 0.0,
      0.0,    0.0,    1.0,    0.0
    };
  }

  bool grab_frame(CameraSide & cam, cv::Mat & frame, const std::string & name) {
    if (!cam.cap.read(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[%s] Empty frame, retrying...", name.c_str());
      return false;
    }

    if (frame.channels() == 1) {
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
      cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    if (frame.cols != width_ || frame.rows != height_) {
      cv::resize(frame, frame, cv::Size(width_, height_));
    }

    return true;
  }

  void publish_frame(CameraSide & cam, const cv::Mat & frame, const rclcpp::Time & stamp) {
    std_msgs::msg::Header hdr;
    hdr.stamp = stamp;
    hdr.frame_id = cam.frame_id;

    cam.img_pub.publish(cv_bridge::CvImage(hdr, "bgr8", frame).toImageMsg());

    cam.cam_info.header.stamp = hdr.stamp;
    cam.cam_info.header.frame_id = cam.frame_id;
    cam.info_pub->publish(cam.cam_info);
  }

  void tick() {
    const auto stamp = now();

    cv::Mat left_frame, right_frame;

    bool left_ok = grab_frame(left_, left_frame, "left");
    bool right_ok = grab_frame(right_, right_frame, "right");

    if (left_ok) {
      publish_frame(left_, left_frame, stamp);
    }
    if (right_ok) {
      publish_frame(right_, right_frame, stamp);
    }
  }

  // Parameters
  double fps_;
  int width_, height_;
  std::string fourcc_;

  // Cameras
  CameraSide left_;
  CameraSide right_;

  // ROS
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DualCamPub>());
  rclcpp::shutdown();
  return 0;
}