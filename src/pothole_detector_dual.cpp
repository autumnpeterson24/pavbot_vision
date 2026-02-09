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
#include <optional>
#include <vector>
#include <cmath>
#include <algorithm>

class PotholeDetectorDual : public rclcpp::Node {
public:
  PotholeDetectorDual()
  : Node("pothole_detector_dual")
  {
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("left_topic",  "/left_cam/image_raw");
    declare_parameter<std::string>("right_topic", "/right_cam/image_raw");
    declare_parameter<std::string>("left_info_topic",  "/left_cam/camera_info");
    declare_parameter<std::string>("right_info_topic", "/right_cam/camera_info");
    declare_parameter<double>("ground_z_in_base", 0.0);

    declare_parameter<double>("roi_top_frac", 0.35);

    // HSV white thresholds
    declare_parameter<int>("sat_max_for_white", 60);
    declare_parameter<int>("val_min_for_white", 220);

    // Morphology
    declare_parameter<int>("open_k", 9);   // kill thin lane lines
    declare_parameter<int>("close_k", 7);  // reconnect disk

    // Contour filters
    declare_parameter<double>("min_area_px", 200.0);
    declare_parameter<double>("max_area_px", 50000.0);
    declare_parameter<double>("min_circularity", 0.65);
    declare_parameter<double>("max_aspect_ratio", 1.6);

    // Expected radius (meters) for output/inflation
    declare_parameter<double>("pothole_radius_m", 0.3048);

    // Fusion
    declare_parameter<double>("fuse_dist_m", 0.40);

    declare_parameter<std::string>("left_cam_frame",  "left_camera_link/left_cam");
    declare_parameter<std::string>("right_cam_frame", "right_camera_link/right_cam");

    //output frame to odom for costmap
    declare_parameter<std::string>("output_frame", "odom");

    declare_parameter<bool>("use_sync_gate", true);
    declare_parameter<double>("sync_slop_sec", 0.05);


    get_parameter("use_sync_gate", use_sync_gate_);
    get_parameter("sync_slop_sec", sync_slop_sec_);

    get_parameter("output_frame", output_frame_);

    get_parameter("left_cam_frame", left_cam_frame_);
    get_parameter("right_cam_frame", right_cam_frame_);


    get_parameter("base_frame", base_frame_);
    get_parameter("left_topic", left_topic_);
    get_parameter("right_topic", right_topic_);
    get_parameter("left_info_topic", left_info_topic_);
    get_parameter("right_info_topic", right_info_topic_);
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
    get_parameter("fuse_dist_m", fuse_dist_m_);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // CameraInfo
    left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      left_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){ K_left_ = parseK(*msg); });

    right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      right_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){ K_right_ = parseK(*msg); });

    // Image subs
    left_sub_ = image_transport::create_subscription(
      this, left_topic_,
      std::bind(&PotholeDetectorDual::onLeft, this, std::placeholders::_1),
      "raw");
    right_sub_ = image_transport::create_subscription(
      this, right_topic_,
      std::bind(&PotholeDetectorDual::onRight, this, std::placeholders::_1),
      "raw");

    poses_pub_   = create_publisher<geometry_msgs::msg::PoseArray>("/potholes/poses", 10);
    radii_pub_   = create_publisher<std_msgs::msg::Float32MultiArray>("/potholes/radii", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/potholes/markers", 10);

    dbg_mask_left_pub_     = image_transport::create_publisher(this, "/potholes/debug_mask_left");
    dbg_mask_right_pub_    = image_transport::create_publisher(this, "/potholes/debug_mask_right");
    dbg_overlay_left_pub_  = image_transport::create_publisher(this, "/potholes/debug_overlay_left");
    dbg_overlay_right_pub_ = image_transport::create_publisher(this, "/potholes/debug_overlay_right");



    RCLCPP_INFO(get_logger(), "pothole_detector_dual up: base_frame=%s", base_frame_.c_str());
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
    std::string cam_frame;
    double u{0}, v{0}; // pixel centroid (full image coords)
    double area{0};
  };

  struct DetBase {
    double x{0}, y{0};
  };

  double minAreaPxAtRange(const KIntr& K, double range_m) const {
  // expected pixel radius ≈ fx * r / range
  const double rp = (K.fx * pothole_r_) / std::max(0.25, range_m);
  const double Ap = M_PI * rp * rp;
  return 0.05 * Ap;  // accept if >= 15% of expected area (tune 0.10–0.25)
}

  bool pixelToGroundBase(const KIntr& K,
                         const std::string& cam_frame,
                         const rclcpp::Time& stamp,
                         double u, double v,
                         double& Xb, double& Yb)
  {
    if (!K.valid) return false;

    const double xr = (u - K.cx) / K.fx;
    const double yr = (v - K.cy) / K.fy;

    // optical -> link mapping (same as your lane node)
    tf2::Vector3 dir_cam(1.0, -xr, -yr);
    dir_cam.normalize();

    geometry_msgs::msg::TransformStamped T;
    try {
          T = tf_buffer_->lookupTransform(
                base_frame_,
                cam_frame,
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
    if (p.x() < 0.0 || p.x() > 50.0) return false;

    Xb = p.x();
    Yb = p.y();
    return true;
  }

  std::vector<Det2D> detectInImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg, const KIntr& K, bool is_left)
    {
    std::vector<Det2D> out;
    if (!K.valid) return out;
    const std::string& cam_frame = is_left ? left_cam_frame_ : right_cam_frame_;


    cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    const int H = bgr.rows;
    const int W = bgr.cols;

    int roi_top = (int)std::round(H * roi_top_frac_);
    roi_top = std::max(0, std::min(roi_top, H - 1));
    cv::Mat roi = bgr.rowRange(roi_top, H);

    // --- HSV white mask ---
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);

    cv::Mat maskS, maskV, mask;
    cv::threshold(ch[1], maskS, sat_max_, 255, cv::THRESH_BINARY_INV);
    cv::threshold(ch[2], maskV, val_min_, 255, cv::THRESH_BINARY);
    cv::bitwise_and(maskS, maskV, mask);

    // --- Morphology ---
    int ok = std::max(1, open_k_ | 1);
    int ck = std::max(1, close_k_ | 1);

    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {ok, ok}));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {ck, ck}));

    // --- Contours ---
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    // Debug overlay: start from ROI BGR
    cv::Mat overlay = roi.clone();

    // Draw all contours lightly (blue)
    if (!contours.empty()) {
        cv::drawContours(overlay, contours, -1, cv::Scalar(255, 0, 0), 1);
    }

    // --- Filter contours, create detections, draw accepted ones (green) ---
    for (const auto& c : contours) {
      double area = cv::contourArea(c);
      if (area > max_area_) continue;   // keep this, remove min_area_ gate

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

      // Full-image pixel coords (important for ray/ground projection)
      double u_full = std::clamp(cx, 0.0, (double)(roi.cols - 1));
      double v_full = std::clamp((double)roi_top + cy, 0.0, (double)(H - 1));

      // Compute range first
      double Xb = 0.0, Yb = 0.0;
      double range_m = 999.0;
      if (pixelToGroundBase(K, cam_frame, msg->header.stamp, u_full, v_full, Xb, Yb)) {
        range_m = std::hypot(Xb, Yb);
      }


      // Dynamic area gate (THIS is the big one)
      double min_area_dyn = minAreaPxAtRange(K, range_m);


      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "range=%.2f area=%.1f min_area_dyn=%.1f circ=%.2f ar=%.2f",
      range_m, area, min_area_dyn, circ, ar);


      if (area < std::max(min_area_, min_area_dyn)) continue;  // not using dynamic right now

      // Dynamic shape gates (after range)
      double min_circ_dyn = min_circ_;
      double max_ar_dyn   = max_ar_;
      if (range_m > 6.0)  { min_circ_dyn = 0.45; max_ar_dyn = 2.2; }
      if (range_m > 10.0) { min_circ_dyn = 0.35; max_ar_dyn = 3.0; }

      if (circ < min_circ_dyn) continue;
      if (ar > max_ar_dyn) continue;

      // Draw accepted bbox + centroid
      cv::rectangle(overlay, bb, cv::Scalar(0, 255, 0), 2);
      cv::circle(overlay, cv::Point((int)cx, (int)cy), 4, cv::Scalar(0, 255, 0), -1);

      Det2D d;
      d.stamp = msg->header.stamp;
      d.cam_frame = cam_frame;
      d.u = u_full;
      d.v = v_full;
      d.area = area;
      out.push_back(d);
    }


    // Debug mask publish (mono8)
    auto& mask_pub    = is_left ? dbg_mask_left_pub_    : dbg_mask_right_pub_;
    auto& overlay_pub = is_left ? dbg_overlay_left_pub_ : dbg_overlay_right_pub_;

    if (mask_pub.getNumSubscribers() > 0) {
    auto mmsg = cv_bridge::CvImage(msg->header, "mono8", mask).toImageMsg();
    mask_pub.publish(mmsg);
    }

    if (overlay_pub.getNumSubscribers() > 0) {
    auto omsg = cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg();
    overlay_pub.publish(omsg);
    }



    return out;
    }

  static double hypot2(double ax, double ay, double bx, double by) {
    return std::hypot(ax - bx, ay - by);
  }

void fuseAndPublish(const rclcpp::Time& stamp)
    {
      // Gather latest per-camera detections and project to base
      std::vector<DetBase> base_pts;

      auto project_list = [&](const std::vector<Det2D>& dets, const KIntr& K) {
        for (const auto& d : dets) {
          double Xb, Yb;
          if (!pixelToGroundBase(K, d.cam_frame, d.stamp, d.u, d.v, Xb, Yb)) continue;

          // // optional wedge gate
          // if (Xb < 2.5 || Xb > 15.0) continue;
          // if (std::abs(Yb) > 3.0) continue;

          base_pts.push_back({Xb, Yb});
        }
      };

      std::vector<Det2D> L, R;
      {
        std::lock_guard<std::mutex> lk(m_);
        L = last_left_;
        R = last_right_;
      }
      project_list(L, K_left_);
      project_list(R, K_right_);

      // Simple clustering in base frame (greedy)
      std::vector<DetBase> fused;
      for (const auto& p : base_pts) {
        bool merged = false;
        for (auto& q : fused) {
          if (hypot2(p.x, p.y, q.x, q.y) < fuse_dist_m_) {
            q.x = 0.5 * (q.x + p.x);
            q.y = 0.5 * (q.y + p.y);
            merged = true;
            break;
          }
        }
        if (!merged) fused.push_back(p);
      }

      // ===============================
      // Tracking helpers (CORRECT SCOPE)
      // ===============================
      const rclcpp::Time now = stamp;

      // expire old tracks once per cycle
      tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
          [&](const Track& t){ return (now - t.last).seconds() > track_ttl_sec_; }),
        tracks_.end());

      // update/create one track from a measurement in output_frame (odom)
      auto update_track = [&](double mx, double my) {
        int best_i = -1;
        double best_d = 1e9;

        for (int i = 0; i < (int)tracks_.size(); ++i) {
          double d = std::hypot(mx - tracks_[i].x, my - tracks_[i].y);
          if (d < best_d) { best_d = d; best_i = i; }
        }

        if (best_i >= 0 && best_d < track_match_dist_) {
          if (best_d > max_jump_m_) return;   // reject insane jump
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
      };

      // ============================================
      // Update tracks from fused measurements (odom)
      // ============================================
      for (size_t i = 0; i < fused.size(); ++i) {

        geometry_msgs::msg::PointStamped p_base;
        p_base.header.frame_id = base_frame_;
        p_base.header.stamp = stamp;
        p_base.point.x = fused[i].x;
        p_base.point.y = fused[i].y;
        p_base.point.z = 0.0;

        geometry_msgs::msg::PointStamped p_out;
        try {
          p_out = tf_buffer_->transform(
            p_base,
            output_frame_,
            tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & ex) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                              "TF transform %s -> %s failed: %s",
                              base_frame_.c_str(), output_frame_.c_str(), ex.what());
          continue;
        }

        update_track(p_out.point.x, p_out.point.y);
      }

      // ============================================
      // Publish stable tracks only (declare ONCE)
      // ============================================
      geometry_msgs::msg::PoseArray poses;
      poses.header.frame_id = output_frame_;
      poses.header.stamp = stamp;

      std_msgs::msg::Float32MultiArray radii;
      visualization_msgs::msg::MarkerArray ma;

      // wipe old markers
      visualization_msgs::msg::Marker wipe;
      wipe.header.frame_id = output_frame_;
      wipe.header.stamp = stamp;
      wipe.ns = "potholes";
      wipe.id = 0;
      wipe.action = visualization_msgs::msg::Marker::DELETEALL;

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


  void maybeFusePublish(const rclcpp::Time& stamp) {
    if (!use_sync_gate_) {
      fuseAndPublish(stamp);
      return;
    }

    rclcpp::Time ls, rs;
    bool hl, hr;
    {
      std::lock_guard<std::mutex> lk(m_);
      ls = last_left_stamp_;
      rs = last_right_stamp_;
      hl = have_left_;
      hr = have_right_;
    }
    if (!(hl && hr)) return;

    const double dt = std::abs((ls - rs).seconds());
    if (dt > sync_slop_sec_) return;

    // Use the newer stamp for publishing
    fuseAndPublish((ls > rs) ? ls : rs);
  }


  void onLeft(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto dets = detectInImage(msg, K_left_, true);
    {
      std::lock_guard<std::mutex> lk(m_);
      last_left_ = std::move(dets);
      last_left_stamp_ = msg->header.stamp;
      have_left_ = true;
    }
    maybeFusePublish(msg->header.stamp);
  }

  void onRight(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto dets = detectInImage(msg, K_right_, false);
    {
      std::lock_guard<std::mutex> lk(m_);
      last_right_ = std::move(dets);
      last_right_stamp_ = msg->header.stamp;
      have_right_ = true;
    }
    maybeFusePublish(msg->header.stamp);
  }



  // Params
  std::string base_frame_;
  std::string left_topic_, right_topic_, left_info_topic_, right_info_topic_;

  std::string left_cam_frame_;
  std::string right_cam_frame_;

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
  double fuse_dist_m_{0.40};

  // K + TF
  KIntr K_left_, K_right_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_, right_info_sub_;
  image_transport::Subscriber left_sub_, right_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr radii_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

    image_transport::Publisher dbg_mask_left_pub_;
    image_transport::Publisher dbg_mask_right_pub_;
    image_transport::Publisher dbg_overlay_left_pub_;
    image_transport::Publisher dbg_overlay_right_pub_;
  std::string output_frame_;

  // Sync gate params
  bool use_sync_gate_{true};
  double sync_slop_sec_{0.05};

  rclcpp::Time last_left_stamp_{0,0,RCL_ROS_TIME};
  rclcpp::Time last_right_stamp_{0,0,RCL_ROS_TIME};
  bool have_left_{false}, have_right_{false};





  std::mutex m_;
  std::vector<Det2D> last_left_, last_right_;

  // try to stop teleporting potholes
  struct Track {
  double x{0}, y{0};
  rclcpp::Time last;
  int hits{0};
  int id{0};
};

std::vector<Track> tracks_;
int next_track_id_{0};

double track_match_dist_{0.75};
double track_alpha_{0.08};
double track_ttl_sec_{2.0};
int    min_hits_to_publish_{2};
double max_jump_m_{0.60};





};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PotholeDetectorDual>());
  rclcpp::shutdown();
  return 0;
}