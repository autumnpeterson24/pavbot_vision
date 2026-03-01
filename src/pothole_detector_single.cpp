#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <cmath>
#include <algorithm>

class PotholeDetectorSingle : public rclcpp::Node {
public:
  PotholeDetectorSingle()
  : Node("pothole_detector_single")
  {
    // Frames / topics
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("image_topic", "/left_cam/image_raw");
    declare_parameter<std::string>("info_topic",  "/left_cam/camera_info");
    declare_parameter<std::string>("cam_frame",   "left_camera_link/left_cam");
    declare_parameter<std::string>("output_frame","odom");
    declare_parameter<double>("ground_z_in_base", 0.0);

    // ROI + thresholds
    declare_parameter<double>("roi_top_frac", 0.35);
    declare_parameter<int>("sat_max_for_white", 60);
    declare_parameter<int>("val_min_for_white", 220);

    // Morphology
    declare_parameter<int>("open_k", 9);
    declare_parameter<int>("close_k", 7);

    // Contour filters
    declare_parameter<double>("min_area_px", 200.0);
    declare_parameter<double>("max_area_px", 50000.0);
    declare_parameter<double>("min_circularity", 0.65);
    declare_parameter<double>("max_aspect_ratio", 1.6);

    // Radius (meters)
    declare_parameter<double>("pothole_radius_m", 0.3048);

    // Optional: within-camera clustering distance (in base frame)
    declare_parameter<double>("cluster_dist_m", 0.35);

    // Tracking / stability
    declare_parameter<double>("track_match_dist", 0.75);
    declare_parameter<double>("track_alpha", 0.08);
    declare_parameter<double>("track_ttl_sec", 2.0);
    declare_parameter<int>("min_hits_to_publish", 2);
    declare_parameter<double>("max_jump_m", 0.60);

    get_parameter("base_frame", base_frame_);
    get_parameter("image_topic", image_topic_);
    get_parameter("info_topic", info_topic_);
    get_parameter("cam_frame", cam_frame_);
    get_parameter("output_frame", output_frame_);
    get_parameter("ground_z_in_base", ground_z_in_base_);

    get_parameter("roi_top_frac", roi_top_frac_);
    get_parameter("sat_max_for_white", sat_max_);
    get_parameter("val_min_for_white", val_min_);
    get_parameter("open_k", open_k_);
    get_parameter("close_k", close_k_);

    get_parameter("min_area_px", min_area_);
    get_parameter("max_area_px", max_area_);
    get_parameter("min_circularity", min_circ_);
    get_parameter("max_aspect_ratio", max_ar_);

    get_parameter("pothole_radius_m", pothole_r_);
    get_parameter("cluster_dist_m", cluster_dist_m_);

    get_parameter("track_match_dist", track_match_dist_);
    get_parameter("track_alpha", track_alpha_);
    get_parameter("track_ttl_sec", track_ttl_sec_);
    get_parameter("min_hits_to_publish", min_hits_to_publish_);
    get_parameter("max_jump_m", max_jump_m_);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){ K_ = parseK(*msg); });

    img_sub_ = image_transport::create_subscription(
      this, image_topic_,
      std::bind(&PotholeDetectorSingle::onImage, this, std::placeholders::_1),
      "raw");

    poses_pub_   = create_publisher<geometry_msgs::msg::PoseArray>("/potholes/poses", 10);
    radii_pub_   = create_publisher<std_msgs::msg::Float32MultiArray>("/potholes/radii", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/potholes/markers", 10);

    dbg_mask_pub_    = image_transport::create_publisher(this, "/potholes/debug_mask");
    dbg_overlay_pub_ = image_transport::create_publisher(this, "/potholes/debug_overlay");

    RCLCPP_INFO(get_logger(),
      "pothole_detector_single up: image=%s info=%s cam_frame=%s base=%s out=%s",
      image_topic_.c_str(), info_topic_.c_str(), cam_frame_.c_str(),
      base_frame_.c_str(), output_frame_.c_str());
  }

private:
  struct KIntr {
    double fx{0}, fy{0}, cx{0}, cy{0};
    int w{0}, h{0};
    bool valid{false};
  };

  static KIntr parseK(const sensor_msgs::msg::CameraInfo& ci) {
    KIntr k;
    k.w = (int)ci.width; k.h = (int)ci.height;
    if (ci.k[0] <= 1e-9 || ci.k[4] <= 1e-9) return k;
    k.fx = ci.k[0]; k.fy = ci.k[4]; k.cx = ci.k[2]; k.cy = ci.k[5];
    k.valid = true;
    return k;
  }

  struct Det2D {
    rclcpp::Time stamp;
    double u{0}, v{0}; // pixel centroid
    double area{0};
  };

  struct Pt2 {
    double x{0}, y{0};
  };

  double minAreaPxAtRange(const KIntr& K, double range_m) const {
    const double rp = (K.fx * pothole_r_) / std::max(0.25, range_m);
    const double Ap = M_PI * rp * rp;
    return 0.05 * Ap; // tune 0.05–0.25
  }

  bool pixelToGroundBase(const KIntr& K,
                         const rclcpp::Time& stamp,
                         double u, double v,
                         double& Xb, double& Yb)
  {
    if (!K.valid) return false;

    const double xr = (u - K.cx) / K.fx;
    const double yr = (v - K.cy) / K.fy;

    // optical -> link mapping (matching your lane node convention)
    tf2::Vector3 dir_cam(1.0, -xr, -yr);
    dir_cam.normalize();

    geometry_msgs::msg::TransformStamped T;
    try {
      T = tf_buffer_->lookupTransform(
        base_frame_,
        cam_frame_,
        stamp,
        rclcpp::Duration::from_seconds(0.05));
    } catch (...) {
      return false;
    }

    tf2::Transform tf_base_from_cam;
    tf2::fromMsg(T.transform, tf_base_from_cam);

    const tf2::Vector3 o_base = tf_base_from_cam.getOrigin();
    const tf2::Vector3 d_base = tf_base_from_cam.getBasis() * dir_cam;

    const double eps = 1e-6;
    if (std::abs(d_base.z()) < eps) return false;

    const double s = (ground_z_in_base_ - o_base.z()) / d_base.z();
    if (!(s > 0.0)) return false;

    const tf2::Vector3 p = o_base + s * d_base;

    // sanity: forward only, and keep your max range
    if (p.x() < 0.0 || p.x() > 50.0) return false;

    Xb = p.x();
    Yb = p.y();
    return true;
  }

  static double dist2(const Pt2& a, const Pt2& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
  }

  std::vector<Det2D> detectInImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    std::vector<Det2D> out;
    if (!K_.valid) return out;

    cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    const int H = bgr.rows;
    const int W = bgr.cols;

    int roi_top = (int)std::round(H * roi_top_frac_);
    roi_top = std::max(0, std::min(roi_top, H - 1));
    cv::Mat roi = bgr.rowRange(roi_top, H);

    // HSV white mask
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);

    cv::Mat maskS, maskV, mask;
    cv::threshold(ch[1], maskS, sat_max_, 255, cv::THRESH_BINARY_INV);
    cv::threshold(ch[2], maskV, val_min_, 255, cv::THRESH_BINARY);
    cv::bitwise_and(maskS, maskV, mask);

    // Morphology
    int ok = std::max(1, open_k_ | 1);
    int ck = std::max(1, close_k_ | 1);

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
      cv::getStructuringElement(cv::MORPH_ELLIPSE, {ok, ok}));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
      cv::getStructuringElement(cv::MORPH_ELLIPSE, {ck, ck}));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat overlay = roi.clone();
    if (!contours.empty()) {
      cv::drawContours(overlay, contours, -1, cv::Scalar(255, 0, 0), 1);
    }

    for (const auto& c : contours) {
      double area = cv::contourArea(c);
      if (area > max_area_) continue;

      double per = cv::arcLength(c, true);
      if (per <= 1e-6) continue;

      cv::Moments m = cv::moments(c);
      if (std::abs(m.m00) < 1e-6) continue;

      double cx = m.m10 / m.m00;
      double cy = m.m01 / m.m00;

      cv::Rect bb = cv::boundingRect(c);

      double ar = (bb.height > 0) ? (double)bb.width / (double)bb.height : 99.0;
      ar = (ar < 1.0) ? 1.0 / ar : ar;

      double circ = 4.0 * M_PI * area / (per * per);

      // full image coords
      double u_full = std::clamp(cx, 0.0, (double)(roi.cols - 1));
      double v_full = std::clamp((double)roi_top + cy, 0.0, (double)(H - 1));

      // range-based min area gate
      double Xb = 0.0, Yb = 0.0;
      double range_m = 999.0;
      if (pixelToGroundBase(K_, msg->header.stamp, u_full, v_full, Xb, Yb)) {
        range_m = std::hypot(Xb, Yb);
      }
      double min_area_dyn = minAreaPxAtRange(K_, range_m);

      if (area < std::max(min_area_, min_area_dyn)) continue;

      // loosen shape far away (same logic as you had)
      double min_circ_dyn = min_circ_;
      double max_ar_dyn   = max_ar_;
      if (range_m > 6.0)  { min_circ_dyn = 0.45; max_ar_dyn = 2.2; }
      if (range_m > 10.0) { min_circ_dyn = 0.35; max_ar_dyn = 3.0; }

      if (circ < min_circ_dyn) continue;
      if (ar > max_ar_dyn) continue;

      cv::rectangle(overlay, bb, cv::Scalar(0, 255, 0), 2);
      cv::circle(overlay, cv::Point((int)cx, (int)cy), 4, cv::Scalar(0, 255, 0), -1);

      out.push_back(Det2D{msg->header.stamp, u_full, v_full, area});
    }

    if (dbg_mask_pub_.getNumSubscribers() > 0) {
      auto mmsg = cv_bridge::CvImage(msg->header, "mono8", mask).toImageMsg();
      dbg_mask_pub_.publish(mmsg);
    }

    if (dbg_overlay_pub_.getNumSubscribers() > 0) {
      auto omsg = cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg();
      dbg_overlay_pub_.publish(omsg);
    }

    return out;
  }

  // -------- tracking ----------
  struct Track {
    double x{0}, y{0};           // in output_frame
    rclcpp::Time last;
    int hits{0};
    int id{0};
  };

  void update_track(const rclcpp::Time& now, double mx, double my) {
    int best_i = -1;
    double best_d = 1e9;

    for (int i = 0; i < (int)tracks_.size(); ++i) {
      double d = std::hypot(mx - tracks_[i].x, my - tracks_[i].y);
      if (d < best_d) { best_d = d; best_i = i; }
    }

    if (best_i >= 0 && best_d < track_match_dist_) {
      if (best_d > max_jump_m_) return;
      auto &t = tracks_[best_i];
      t.x = (1.0 - track_alpha_) * t.x + track_alpha_ * mx;
      t.y = (1.0 - track_alpha_) * t.y + track_alpha_ * my;
      t.last = now;
      t.hits++;
    } else {
      Track t;
      t.x = mx; t.y = my;
      t.last = now;
      t.hits = 1;
      t.id = next_track_id_++;
      tracks_.push_back(t);
    }
  }

  void publish_tracks(const rclcpp::Time& stamp) {
    geometry_msgs::msg::PoseArray poses;
    poses.header.frame_id = output_frame_;
    poses.header.stamp = stamp;

    std_msgs::msg::Float32MultiArray radii;
    visualization_msgs::msg::MarkerArray ma;

    visualization_msgs::msg::Marker wipe;
    wipe.header = poses.header;
    wipe.ns = "potholes";
    wipe.id = 0;
    wipe.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(wipe);

    for (const auto& t : tracks_) {
      if (t.hits < min_hits_to_publish_) continue;

      geometry_msgs::msg::Pose pose;
      pose.position.x = t.x;
      pose.position.y = t.y;
      pose.position.z = 0.0;
      pose.orientation.w = 1.0;

      poses.poses.push_back(pose);
      radii.data.push_back((float)pothole_r_);

      visualization_msgs::msg::Marker m;
      m.header = poses.header;
      m.ns = "potholes";
      m.id = t.id;
      m.type = visualization_msgs::msg::Marker::CYLINDER;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose = pose;
      m.scale.x = 2.0 * pothole_r_;
      m.scale.y = 2.0 * pothole_r_;
      m.scale.z = 0.02;
      m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 0.6;
      ma.markers.push_back(m);
    }

    poses_pub_->publish(poses);
    radii_pub_->publish(radii);
    markers_pub_->publish(ma);
  }

  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto dets = detectInImage(msg);
    if (!K_.valid) return;

    // project all detections into base frame
    std::vector<Pt2> pts_base;
    pts_base.reserve(dets.size());

    for (const auto& d : dets) {
      double Xb, Yb;
      if (!pixelToGroundBase(K_, d.stamp, d.u, d.v, Xb, Yb)) continue;
      pts_base.push_back({Xb, Yb});
    }

    // optional clustering to reduce duplicates within the same image
    std::vector<Pt2> clustered;
    for (const auto& p : pts_base) {
      bool merged = false;
      for (auto& q : clustered) {
        if (dist2(p, q) < cluster_dist_m_) {
          q.x = 0.5 * (q.x + p.x);
          q.y = 0.5 * (q.y + p.y);
          merged = true;
          break;
        }
      }
      if (!merged) clustered.push_back(p);
    }

    // expire old tracks
    const rclcpp::Time now = msg->header.stamp;
    tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(),
        [&](const Track& t){ return (now - t.last).seconds() > track_ttl_sec_; }),
      tracks_.end());

    // update tracks using output_frame
    for (const auto& p : clustered) {
      geometry_msgs::msg::PointStamped p_base;
      p_base.header.frame_id = base_frame_;
      p_base.header.stamp = msg->header.stamp;
      p_base.point.x = p.x;
      p_base.point.y = p.y;
      p_base.point.z = 0.0;

      geometry_msgs::msg::PointStamped p_out;
      try {
        p_out = tf_buffer_->transform(p_base, output_frame_, tf2::durationFromSec(0.05));
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
          "TF transform %s -> %s failed: %s",
          base_frame_.c_str(), output_frame_.c_str(), ex.what());
        continue;
      }

      update_track(now, p_out.point.x, p_out.point.y);
    }

    publish_tracks(msg->header.stamp);
  }

  // Params
  std::string base_frame_;
  std::string image_topic_, info_topic_;
  std::string cam_frame_;
  std::string output_frame_;
  double ground_z_in_base_{0.0};

  double roi_top_frac_{0.35};
  int sat_max_{60};
  int val_min_{220};
  int open_k_{9};
  int close_k_{7};

  double min_area_{200.0};
  double max_area_{50000.0};
  double min_circ_{0.65};
  double max_ar_{1.6};

  double pothole_r_{0.3048};
  double cluster_dist_m_{0.35};

  // Tracking params
  double track_match_dist_{0.75};
  double track_alpha_{0.08};
  double track_ttl_sec_{2.0};
  int    min_hits_to_publish_{2};
  double max_jump_m_{0.60};

  // Intrinsics + TF
  KIntr K_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  image_transport::Subscriber img_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr radii_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

  image_transport::Publisher dbg_mask_pub_;
  image_transport::Publisher dbg_overlay_pub_;

  // Tracks
  std::vector<Track> tracks_;
  int next_track_id_{0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PotholeDetectorSingle>());
  rclcpp::shutdown();
  return 0;
}