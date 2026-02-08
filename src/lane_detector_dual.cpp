#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <opencv2/opencv.hpp>

#include <mutex>
#include <optional>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

//new ray-ground
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>



struct Poly2 {
  double a{0}, b{0}, c{0};
  bool valid{false};
  double eval(double y) const { return a*y*y + b*y + c; }
};

// -----------------------------
// Helpers
// -----------------------------
static inline float clamp01(float v) {
  return std::max(0.0f, std::min(1.0f, v));
}

static inline float smoothstep(float edge0, float edge1, float x) {
  if (edge1 <= edge0) return (x >= edge1) ? 1.0f : 0.0f;
  float t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3.0f - 2.0f * t);
}

static inline double clampd(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

static inline double exp_smooth_alpha_from_dt(double dt, double tau) {
  // For time-constant low-pass: alpha = exp(-dt/tau). alpha close to 1 => more smoothing.
  if (!(dt > 0.0) || !(tau > 1e-6)) return 0.0;  // 0 => no smoothing (use new)
  double a = std::exp(-dt / tau);
  return clampd(a, 0.0, 0.995);
}

static Poly2 fit_poly2(const std::vector<cv::Point>& pts) {
  Poly2 poly;
  if (pts.size() < 15) return poly;

  cv::Mat A((int)pts.size(), 3, CV_64F), X((int)pts.size(), 1, CV_64F);
  for (size_t i = 0; i < pts.size(); ++i) {
    const double y = (double)pts[i].y;
    const double x = (double)pts[i].x;
    A.at<double>((int)i,0) = y*y;
    A.at<double>((int)i,1) = y;
    A.at<double>((int)i,2) = 1.0;
    X.at<double>((int)i,0) = x;
  }

  cv::Mat coef;
  if (!cv::solve(A, X, coef, cv::DECOMP_SVD)) return poly;

  poly.a = coef.at<double>(0,0);
  poly.b = coef.at<double>(1,0);
  poly.c = coef.at<double>(2,0);
  poly.valid = true;
  return poly;
}

static void draw_poly(cv::Mat& img, const Poly2& p, const cv::Scalar& col) {
  for (int y = img.rows - 1; y >= 0; y -= 5) {
    int x = (int)std::round(p.eval((double)y));
    if (0 <= x && x < img.cols) cv::circle(img, {x, y}, 2, col, -1);
  }
}

// Robust refit: fit once, keep inliers by residual threshold, refit.
static Poly2 robust_refit_poly2(const std::vector<cv::Point>& pts,
                                int min_pts,
                                double inlier_thresh_px,
                                int min_inliers)
{
  Poly2 p0 = fit_poly2(pts);
  if (!p0.valid) return p0;
  if ((int)pts.size() < min_pts) return Poly2{};

  std::vector<cv::Point> inliers;
  inliers.reserve(pts.size());
  for (const auto& pt : pts) {
    const double xhat = p0.eval((double)pt.y);
    if (!std::isfinite(xhat)) continue;
    const double r = std::abs((double)pt.x - xhat);
    if (r <= inlier_thresh_px) inliers.push_back(pt);
  }

  if ((int)inliers.size() < std::max(min_inliers, min_pts)) {
    // Not enough consistent support; mark invalid to prevent high-speed snap.
    return Poly2{};
  }
  return fit_poly2(inliers);
}

static Poly2 blend_poly(const Poly2& prev, const Poly2& curr, double alpha_prev) {
  // alpha_prev in [0..1): out = alpha_prev*prev + (1-alpha_prev)*curr
  if (!curr.valid) return Poly2{};
  if (!prev.valid) return curr;
  Poly2 out;
  out.a = alpha_prev * prev.a + (1.0 - alpha_prev) * curr.a;
  out.b = alpha_prev * prev.b + (1.0 - alpha_prev) * curr.b;
  out.c = alpha_prev * prev.c + (1.0 - alpha_prev) * curr.c;
  out.valid = true;
  return out;
}

struct BoundaryResult {
  Poly2 poly;
  float confidence{0.f};
  cv::Mat debug_bgr;
  int roi_top{0};
  int W{0}, H{0};
  rclcpp::Time stamp;

  // diagnostics
  int num_points{0};
  int windows_hit{0};
  int nwindows{0};
  float mean_residual_px{1e9f};
  float frac_in_expected_half{0.f};
};

class LaneDetectorDualNode : public rclcpp::Node {
public:
  LaneDetectorDualNode() : Node("lane_detector_dual") {



    // NEW INFO TOPICS FOR TF
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("left_camera_info_topic",  "/left_cam/camera_info");
    declare_parameter<std::string>("right_camera_info_topic", "/right_cam/camera_info");

        // new ground base
    declare_parameter<double>("ground_z_in_base", 0.0);
    get_parameter("ground_z_in_base", ground_z_in_base_);


    // Topics / frames
    declare_parameter<std::string>("left_camera_topic",  "/left_cam/image_raw");
    declare_parameter<std::string>("right_camera_topic", "/right_cam/image_raw");
    declare_parameter<std::string>("path_frame", "base_link");

    // Geometry conversion
    declare_parameter<double>("roi_top_frac", 0.3);
    //declare_parameter<double>("mppx", 0.025);
    //declare_parameter<double>("mppy", 0.025);

    // If only one boundary exists
    declare_parameter<double>("nominal_lane_half_width_m", 1.5);

    // Camera offsets relative to base_link (ROS: +y left, -y right)
    //declare_parameter<double>("left_cam_y_offset_m",  0.0);
    //declare_parameter<double>("right_cam_y_offset_m", -0.0);

    // Optional timestamp sync gate
    declare_parameter<bool>("use_sync_gate", false);
    declare_parameter<double>("sync_slop_sec", 0.08);

    // NEW: freshness gating (prevents mixing very stale frames at higher speed)
    declare_parameter<double>("max_boundary_age_sec", 0.20);    // ignore boundary if older than this
    declare_parameter<double>("max_pair_skew_sec", 0.05);       // when both exist, prefer close timestamps

    // Confidence tuning
    declare_parameter<int>("min_points_fit", 15);
    declare_parameter<int>("support_pts_lo", 150);
    declare_parameter<int>("support_pts_hi", 900);
    declare_parameter<double>("residual_good_px", 6.0);
    declare_parameter<double>("residual_bad_px", 18.0);
    declare_parameter<double>("curv_good", 1e-4);
    declare_parameter<double>("curv_bad", 8e-4);

    // NEW: robust refit for higher speeds (reject outliers => less snapping)
    declare_parameter<bool>("robust_refit", true);
    declare_parameter<double>("robust_inlier_thresh_px", 10.0);
    declare_parameter<int>("robust_min_inliers", 40);

    // NEW: temporal smoothing of per-camera polynomials (reduces jitter at speed)
    declare_parameter<double>("poly_smooth_tau_sec", 0.25);  // time constant; lower = more responsive

    // NEW: segmentation knobs (so you can tighten for speed/lighting)
    declare_parameter<int>("sat_max_for_white", 60);
    declare_parameter<int>("val_min_for_white", 200);
    declare_parameter<int>("sobel_thresh", 40);

    // Debug throttling
    declare_parameter<double>("debug_print_hz", 1.0);

    // Fusion robustness
    declare_parameter<double>("min_lane_confidence", 0.35);
    declare_parameter<double>("lane_conf_hysteresis", 0.07);
    declare_parameter<double>("centerline_smooth_alpha", 0.85);

    // Auto-learn lane half width from dual-lane frames
    declare_parameter<bool>("auto_learn_half_width", true);
    declare_parameter<double>("half_width_learn_alpha", 0.90);
    declare_parameter<double>("half_width_min_m", 0.25);
    declare_parameter<double>("half_width_max_m", 2.20);

    // Preference / hold
    declare_parameter<int>("prefer_switch_threshold", 6);
    declare_parameter<double>("center_hold_sec", 0.25);

    declare_parameter<std::string>("left_camera_frame",  "left_camera_link/left_cam");
    declare_parameter<std::string>("right_camera_frame", "right_camera_link/right_cam");
    get_parameter("left_camera_frame",  left_cam_frame_);
    get_parameter("right_camera_frame", right_cam_frame_);




    // Load params

    //NEW PARAMS
    get_parameter("base_frame", base_frame_);
    get_parameter("left_camera_info_topic", left_info_topic_);
    get_parameter("right_camera_info_topic", right_info_topic_);

    get_parameter("left_camera_topic", left_topic_);
    get_parameter("right_camera_topic", right_topic_);
    get_parameter("path_frame", frame_);

    get_parameter("roi_top_frac", roi_top_frac_);
    //get_parameter("mppx", mppx_);
    //get_parameter("mppy", mppy_);

    get_parameter("nominal_lane_half_width_m", nominal_half_width_);
    //get_parameter("left_cam_y_offset_m", left_cam_y_off_);
    //get_parameter("right_cam_y_offset_m", right_cam_y_off_);

    get_parameter("use_sync_gate", use_sync_gate_);
    get_parameter("sync_slop_sec", sync_slop_sec_);
    get_parameter("max_boundary_age_sec", max_boundary_age_sec_);
    get_parameter("max_pair_skew_sec", max_pair_skew_sec_);

    get_parameter("min_points_fit", min_points_fit_);
    get_parameter("support_pts_lo", support_pts_lo_);
    get_parameter("support_pts_hi", support_pts_hi_);
    get_parameter("residual_good_px", residual_good_px_);
    get_parameter("residual_bad_px", residual_bad_px_);
    get_parameter("curv_good", curv_good_);
    get_parameter("curv_bad", curv_bad_);

    get_parameter("robust_refit", robust_refit_);
    get_parameter("robust_inlier_thresh_px", robust_inlier_thresh_px_);
    get_parameter("robust_min_inliers", robust_min_inliers_);
    get_parameter("poly_smooth_tau_sec", poly_smooth_tau_sec_);

    get_parameter("sat_max_for_white", sat_max_for_white_);
    get_parameter("val_min_for_white", val_min_for_white_);
    get_parameter("sobel_thresh", sobel_thresh_);

    get_parameter("min_lane_confidence", min_lane_conf_);
    get_parameter("lane_conf_hysteresis", lane_conf_hyst_);
    get_parameter("centerline_smooth_alpha", center_smooth_alpha_);

    get_parameter("auto_learn_half_width", auto_learn_half_width_);
    get_parameter("half_width_learn_alpha", half_width_learn_alpha_);
    get_parameter("half_width_min_m", half_width_min_m_);
    get_parameter("half_width_max_m", half_width_max_m_);

    get_parameter("prefer_switch_threshold", prefer_switch_threshold_);
    get_parameter("center_hold_sec", center_hold_sec_);

    get_parameter("debug_print_hz", debug_print_hz_);


    debug_print_period_ms_ = (debug_print_hz_ <= 0.0) ? 0 : (int)(1000.0 / debug_print_hz_);


      footprint_pub_ = create_publisher<visualization_msgs::msg::Marker>("/robot_footprint", 10);

      footprint_timer_ = create_wall_timer(
        std::chrono::milliseconds(200),
        [this]() { publishFootprint(); }
      );



    // Publishers
    center_pub_ = create_publisher<nav_msgs::msg::Path>("/lanes/centerline", 10);
    left_pub_   = create_publisher<nav_msgs::msg::Path>("/lanes/left_boundary", 10);
    right_pub_  = create_publisher<nav_msgs::msg::Path>("/lanes/right_boundary", 10);
    conf_pub_   = create_publisher<std_msgs::msg::Float32>("/lanes/confidence", 10);

    dbg_left_pub_  = image_transport::create_publisher(this, "/lanes/debug_left");
    dbg_right_pub_ = image_transport::create_publisher(this, "/lanes/debug_right");

    // Subscribers
    left_sub_ = image_transport::create_subscription(
      this, left_topic_,
      std::bind(&LaneDetectorDualNode::leftCb, this, std::placeholders::_1),
      "raw");

    right_sub_ = image_transport::create_subscription(
      this, right_topic_,
      std::bind(&LaneDetectorDualNode::rightCb, this, std::placeholders::_1),
      "raw");

    // TF buffer/listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // CameraInfo subs
    left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      left_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){
        K_left_ = parseK(*msg);
      });

    right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      right_info_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg){
        K_right_ = parseK(*msg);
      });

    RCLCPP_INFO(get_logger(), "lane_detector_dual up.");
    RCLCPP_INFO(get_logger(), "  left : %s", left_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  right: %s", right_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  frame: %s", frame_.c_str());
    RCLCPP_INFO(get_logger(), "  roi_top_frac=%.2f ", roi_top_frac_);
    RCLCPP_INFO(get_logger(), "  robust_refit=%s  poly_tau=%.3fs  max_age=%.3fs",
                robust_refit_ ? "true" : "false", poly_smooth_tau_sec_, max_boundary_age_sec_);
  }

private:
  // -----------------------------
  // Fusion params
  // -----------------------------
  double min_lane_conf_{0.35};
  double lane_conf_hyst_{0.07};
  double center_smooth_alpha_{0.85};

  bool auto_learn_half_width_{true};
  double half_width_learn_alpha_{0.90};
  double half_width_min_m_{0.25};
  double half_width_max_m_{2.20};

  // Freshness / sync
  bool use_sync_gate_{false};
  double sync_slop_sec_{0.08};
  double max_boundary_age_sec_{0.20};
  double max_pair_skew_sec_{0.05};

  // Robust + temporal
  bool robust_refit_{true};
  double robust_inlier_thresh_px_{10.0};
  int robust_min_inliers_{40};
  double poly_smooth_tau_sec_{0.25};

  // Segmentation tuning
  int sat_max_for_white_{60};
  int val_min_for_white_{200};
  int sobel_thresh_{40};

  // Fusion state
  bool left_ok_{false};
  bool right_ok_{false};
  std::optional<nav_msgs::msg::Path> last_centerline_;

  // Preference state
  enum class PreferredSide { LEFT, RIGHT };
  PreferredSide preferred_side_{PreferredSide::LEFT};
  int prefer_switch_count_{0};
  int prefer_switch_threshold_{6};

  // Hold last published centerline briefly
  double center_hold_sec_{0.25};
  rclcpp::Time last_center_pub_time_{0, 0, RCL_ROS_TIME};
  nav_msgs::msg::Path last_center_pub_;
  bool has_last_center_pub_{false};


    // NEW Ray ground transform
    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr footprint_pub_;
    rclcpp::TimerBase::SharedPtr footprint_timer_;



  // Per-side poly filter state
  struct PolyState {
    Poly2 last;
    rclcpp::Time stamp{0,0,RCL_ROS_TIME};
    bool has{false};
  };
  PolyState left_poly_state_;
  PolyState right_poly_state_;

      // CameraInfo
    struct KIntr {
      double fx{0}, fy{0}, cx{0}, cy{0};
      int w{0}, h{0};
      bool valid{false};
    };

    KIntr K_left_, K_right_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;

    std::string base_frame_{"base_link"};
    std::string left_info_topic_{"/left_cam/camera_info"};
    std::string right_info_topic_{"/right_cam/camera_info"};

    std::string left_cam_frame_{"left_camera_link/left_cam"};
    std::string right_cam_frame_{"right_camera_link/right_cam"};

  static KIntr parseK(const sensor_msgs::msg::CameraInfo& ci) {
    KIntr k;
    k.w = (int)ci.width;
    k.h = (int)ci.height;
    if (ci.k[0] <= 1e-9 || ci.k[4] <= 1e-9) return k;
    k.fx = ci.k[0];
    k.fy = ci.k[4];
    k.cx = ci.k[2];
    k.cy = ci.k[5];
    k.valid = true;
    return k;
  }

  void publishFootprint() {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = base_frame_;          // or "base_link"
    m.header.stamp = now();

    m.ns = "footprint";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;

    m.pose.orientation.w = 1.0;

    m.scale.x = 0.05;      // line width

    m.color.r = 1.0;
    m.color.g = 0.0;
    m.color.b = 0.0;
    m.color.a = 1.0;

    geometry_msgs::msg::Point p;
    p.z = 0.0;

    // rectangle footprint (adjust to your robot dimensions)
    p.x = -0.5; p.y = -0.35; m.points.push_back(p);
    p.x =  0.5; p.y = -0.35; m.points.push_back(p);
    p.x =  0.5; p.y =  0.35; m.points.push_back(p);
    p.x = -0.5; p.y =  0.35; m.points.push_back(p);
    p.x = -0.5; p.y = -0.35; m.points.push_back(p);

    footprint_pub_->publish(m);
  }



  // Project pixel (u,v) from camera frame into base_frame ground plane z=0.
// Returns true if intersection is valid and in front of camera.
bool pixelToGroundBase(const KIntr& K,
                       const std::string& cam_frame,
                       const rclcpp::Time& stamp,
                       double u, double v,
                       double& Xb, double& Yb)
{
  if (!K.valid) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "pixelToGroundBase FAIL (K invalid) cam=%s", cam_frame.c_str());
    return false;
  }

  // 1) pixel -> normalized ray in *some* camera frame
  // Image normalized coordinates:
  const double xr = (u - K.cx) / K.fx;   // + right in image
  const double yr = (v - K.cy) / K.fy;   // + down in image

  // ---- Ray convention mapping ----
  // If your TF frame is a typical "camera link" frame (Gazebo often):
  //   +X forward, +Y left, +Z up
  // while image math is "optical":
  //   +X right, +Y down, +Z forward
  //
  // This maps optical -> link:
  //   forward = z_opt = 1
  //   left    = -x_opt = -xr
  //   up      = -y_opt = -yr
  tf2::Vector3 dir_cam(1.0, -xr, -yr);
  dir_cam.normalize();

  // 2) lookup TF: base_frame <- cam_frame at the image timestamp
  geometry_msgs::msg::TransformStamped T;
  try {
    T = tf_buffer_->lookupTransform(base_frame_, cam_frame, stamp,
                                    rclcpp::Duration::from_seconds(0.05));
  } catch (const tf2::TransformException& ex) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "pixelToGroundBase FAIL (TF lookup) base=%s cam=%s err=%s",
      base_frame_.c_str(), cam_frame.c_str(), ex.what());
    return false;
  }

  tf2::Transform tf_base_from_cam;
  tf2::fromMsg(T.transform, tf_base_from_cam);

  // Camera origin in base, and ray direction in base
  const tf2::Vector3 o_base = tf_base_from_cam.getOrigin();
  const tf2::Vector3 d_base = tf_base_from_cam.getBasis() * dir_cam;

  // 3) Intersect with ground plane z=0 in base frame
  const double eps = 1e-6;

  if (std::abs(d_base.z()) < eps) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "pixelToGroundBase FAIL (parallel) cam=%s u=%.1f v=%.1f "
      "o=(%.3f %.3f %.3f) d=(%.3f %.3f %.3f)",
      cam_frame.c_str(), u, v,
      o_base.x(), o_base.y(), o_base.z(),
      d_base.x(), d_base.y(), d_base.z());
    return false;
  }

  const double s = (ground_z_in_base_ - o_base.z()) / d_base.z();

  if (!(s > 0.0)) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "pixelToGroundBase FAIL (behind) cam=%s u=%.1f v=%.1f "
      "s=%.3f o.z=%.3f d.z=%.3f o=(%.3f %.3f %.3f) d=(%.3f %.3f %.3f)",
      cam_frame.c_str(), u, v,
      s, o_base.z(), d_base.z(),
      o_base.x(), o_base.y(), o_base.z(),
      d_base.x(), d_base.y(), d_base.z());
    return false;
  }

  const tf2::Vector3 p = o_base + s * d_base;

  // Optional: reject absurd ranges (helps with sky pixels / bad conventions)
  if (p.x() < 0.0 || p.x() > 50.0) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "pixelToGroundBase FAIL (range) cam=%s u=%.1f v=%.1f "
      "p=(%.3f %.3f %.3f) s=%.3f",
      cam_frame.c_str(), u, v,
      p.x(), p.y(), p.z(), s);
    return false;
  }

  // Success
  Xb = p.x();
  Yb = p.y();

  // Throttled success print (optional but nice during bring-up)
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
    "pixelToGroundBase OK cam=%s u=%.1f v=%.1f -> base (%.3f %.3f) "
    "o=(%.3f %.3f %.3f) d=(%.3f %.3f %.3f) s=%.3f",
    cam_frame.c_str(), u, v, Xb, Yb,
    o_base.x(), o_base.y(), o_base.z(),
    d_base.x(), d_base.y(), d_base.z(),
    s);

  return true;
}



  // -----------------------------
  // Core processing
  // -----------------------------
  BoundaryResult processOneBoundary(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                                   bool is_left_camera)
  {
    BoundaryResult out;
    out.stamp = msg->header.stamp;

    cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    out.W = bgr.cols;

    out.roi_top = (int)std::round((double)bgr.rows * roi_top_frac_);
    out.roi_top = std::max(0, std::min(out.roi_top, bgr.rows - 1));
    cv::Mat roi = bgr.rowRange(out.roi_top, bgr.rows);
    out.H = roi.rows;

    // --- segmentation: "white-ish" + edges ---
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);

    cv::Mat maskS, maskV, mask;
    cv::threshold(ch[1], maskS, sat_max_for_white_, 255, cv::THRESH_BINARY_INV);
    cv::threshold(ch[2], maskV, val_min_for_white_, 255, cv::THRESH_BINARY);
    cv::bitwise_and(maskS, maskV, mask);

    cv::Mat gray, gradx, absx, edges;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, gradx, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(gradx, absx);
    cv::threshold(absx, edges, sobel_thresh_, 255, cv::THRESH_BINARY);

    cv::bitwise_or(mask, edges, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, {3,3}));

    const int H = mask.rows;
    const int W = mask.cols;

    // Histogram search in expected half
    cv::Mat hist;
    cv::reduce(mask.rowRange(H/2, H), hist, 0, cv::REDUCE_SUM, CV_32S);

    int x_base = (is_left_camera ? W/3 : (2*W)/3);
    int x_lo = is_left_camera ? 0 : W/2;
    int x_hi = is_left_camera ? W/2 : W;

    double best = 0.0;
    for (int x = x_lo; x < x_hi; ++x) {
      int v = hist.at<int>(0, x);
      if ((double)v > best) { best = (double)v; x_base = x; }
    }

    // Sliding windows for ONE line
    const int nwindows = 9;
    const int winH = std::max(1, H / nwindows);
    const int margin = 60;
    const int minpix = 50;

    std::vector<cv::Point> P;
    P.reserve(2000);

    int x_curr = x_base;
    int windows_hit = 0;

    for (int w = 0; w < nwindows; ++w) {
      int ylo = H - (w + 1) * winH;
      int yhi = H - w * winH;
      if (ylo < 0) ylo = 0;
      if (yhi > H) yhi = H;
      if (yhi <= ylo) continue;

      int xlo2 = std::max(0, x_curr - margin);
      int xhi2 = std::min(W - 1, x_curr + margin);
      if (xhi2 < xlo2) continue;

      cv::Rect win(xlo2, ylo, (xhi2 - xlo2 + 1), (yhi - ylo));
      std::vector<cv::Point> nz;
      cv::findNonZero(mask(win), nz);

      for (auto &p : nz) P.emplace_back(p.x + xlo2, p.y + ylo);

      if ((int)nz.size() > minpix) {
        windows_hit++;
        int s = 0;
        for (auto &p : nz) s += (p.x + xlo2);
        x_curr = s / (int)nz.size();
      }
    }

    out.num_points = (int)P.size();
    out.windows_hit = windows_hit;
    out.nwindows = nwindows;

    // Fit (robust, then temporal smooth)
    Poly2 fitted;
    if (robust_refit_) {
      fitted = robust_refit_poly2(P, min_points_fit_, robust_inlier_thresh_px_, robust_min_inliers_);
    } else {
      fitted = fit_poly2(P);
    }

    // Temporal smoothing of the poly coefficients (reduces high-speed jitter)
    if (fitted.valid) {
      PolyState& st = is_left_camera ? left_poly_state_ : right_poly_state_;
      double dt = st.has ? (out.stamp - st.stamp).seconds() : 0.0;
      double a_prev = st.has ? exp_smooth_alpha_from_dt(dt, poly_smooth_tau_sec_) : 0.0;
      Poly2 blended = blend_poly(st.has ? st.last : Poly2{}, fitted, a_prev);
      st.last = blended;
      st.stamp = out.stamp;
      st.has = true;
      out.poly = blended;
    } else {
      out.poly = Poly2{};
      // If we failed, do not update the filter state; keep previous for stability.
    }

    // --- confidence ---
    if (!out.poly.valid || (int)P.size() < min_points_fit_) {
      out.confidence = 0.0f;
      out.mean_residual_px = 1e9f;
      out.frac_in_expected_half = 0.0f;
    } else {
      // A) Support
      float support = smoothstep((float)support_pts_lo_, (float)support_pts_hi_, (float)P.size());

      // B) Continuity
      float continuity_raw = (float)windows_hit / (float)nwindows;
      float continuity = smoothstep(0.35f, 0.85f, continuity_raw);

      // C) Fit residual + expected-half sanity
      const int x_expected_lo = is_left_camera ? 0 : W/2;
      const int x_expected_hi = is_left_camera ? W/2 : W;

      double sum_abs = 0.0;
      int n = 0;
      int in_expected = 0;

      for (const auto &pt : P) {
        const double xhat = out.poly.eval((double)pt.y);
        if (!std::isfinite(xhat)) continue;
        sum_abs += std::abs((double)pt.x - xhat);
        n++;
        if (pt.x >= x_expected_lo && pt.x < x_expected_hi) in_expected++;
      }

      float mean_res = (n > 0) ? (float)(sum_abs / (double)n) : 1e9f;
      out.mean_residual_px = mean_res;

      float fitq = 1.0f - smoothstep((float)residual_good_px_, (float)residual_bad_px_, mean_res);

      float frac_expected = (n > 0) ? (float)in_expected / (float)n : 0.0f;
      out.frac_in_expected_half = frac_expected;
      float side = smoothstep(0.60f, 0.90f, frac_expected);

      // E) Curvature sanity
      float curv = 1.0f - smoothstep((float)curv_good_, (float)curv_bad_, (float)std::abs(out.poly.a));

      float conf = 0.35f * support
                 + 0.25f * continuity
                 + 0.25f * fitq
                 + 0.10f * side
                 + 0.05f * curv;

      out.confidence = clamp01(conf);
    }

    // Debug image
    cv::Mat dbg;
    cv::cvtColor(mask, dbg, cv::COLOR_GRAY2BGR);
    if (out.poly.valid) draw_poly(dbg, out.poly, cv::Scalar(0,255,0));
    out.debug_bgr = dbg;

    return out;
  }

nav_msgs::msg::Path boundaryToPathProjected(const BoundaryResult& r,
                                           const KIntr& K,
                                           const std::string& cam_frame)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = base_frame_;
  path.header.stamp = r.stamp;

  // Basic guards
  if (!r.poly.valid || r.W <= 0 || r.H <= 0) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "boundaryToPathProjected: poly_invalid_or_bad_dims cam=%s poly=%d W=%d H=%d roi_top=%d pts=%d",
      cam_frame.c_str(), (int)r.poly.valid, r.W, r.H, r.roi_top, r.num_points);
    return path;
  }
  if (!K.valid) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "boundaryToPathProjected: K invalid cam=%s (no camera_info yet?)",
      cam_frame.c_str());
    return path;
  }

  const double roi_y0 = (double)r.roi_top;

  int n_try = 0;
  int n_ok  = 0;

  // Sample along the fitted poly in ROI pixel coords, bottom -> top
  for (int y_roi = r.H - 1; y_roi >= 0; y_roi -= 5) {
    n_try++;

    const double x_roi = r.poly.eval((double)y_roi);
    if (!std::isfinite(x_roi)) continue;

    // Convert ROI pixel -> full image pixel (u,v)
    const double u = clampd(x_roi, 0.0, (double)(r.W - 1));
    const double v = clampd(roi_y0 + (double)y_roi, 0.0, (double)(r.roi_top + r.H - 1));

    double Xb = 0.0, Yb = 0.0;
    if (!pixelToGroundBase(K, cam_frame, r.stamp, u, v, Xb, Yb)) {
      continue;
    }

    // Optional sanity (prevents a few bad rays from polluting the path)
    if (!std::isfinite(Xb) || !std::isfinite(Yb)) continue;
    if (Xb < 0.0 || Xb > 50.0) continue;

    geometry_msgs::msg::PoseStamped p;
    p.header = path.header;
    p.pose.position.x = Xb;
    p.pose.position.y = Yb;
    p.pose.position.z = 0.0;
    p.pose.orientation.w = 1.0;
    path.poses.push_back(p);
    n_ok++;
  }

  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
    "project cam=%s -> base=%s : try=%d ok=%d poly=%d pts=%d roi_top=%d W=%d H=%d K=%d",
    cam_frame.c_str(), base_frame_.c_str(),
    n_try, n_ok, (int)r.poly.valid, r.num_points,
    r.roi_top, r.W, r.H, (int)K.valid
  );

  return path;
}



  // Robust median half-width estimate
  double estimateHalfWidthMedian(const nav_msgs::msg::Path& left,
                                 const nav_msgs::msg::Path& right) {
    size_t N = std::min(left.poses.size(), right.poses.size());
    if (N < 8) return nominal_half_width_;

    std::vector<double> hw;
    hw.reserve(N);
    for (size_t i = 0; i < N; ++i) {
      double dy = std::abs(left.poses[i].pose.position.y - right.poses[i].pose.position.y);
      hw.push_back(0.5 * dy);
    }
    std::nth_element(hw.begin(), hw.begin() + hw.size()/2, hw.end());
    double med = hw[hw.size()/2];

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "halfwidth debug: N=%zu  med_half=%.3f  -> med_dy=%.3f",
      N, med, 2.0*med
    );


    return clampd(med, half_width_min_m_, half_width_max_m_);
  }

  // Smooth centerline (y only) to prevent snapping/jitter
  nav_msgs::msg::Path smoothCenterline(const nav_msgs::msg::Path& in) {
    if (!last_centerline_ || last_centerline_->poses.empty() || in.poses.empty()) {
      last_centerline_ = in;
      return in;
    }

    nav_msgs::msg::Path out = in;
    size_t N = std::min(out.poses.size(), last_centerline_->poses.size());
    const double a = clampd(center_smooth_alpha_, 0.0, 0.99);

    for (size_t i = 0; i < N; ++i) {
      const double y_prev = last_centerline_->poses[i].pose.position.y;
      const double y_new  = out.poses[i].pose.position.y;
      out.poses[i].pose.position.y = a * y_prev + (1.0 - a) * y_new;
    }

    last_centerline_ = out;
    return out;
  }

  bool updateUsable(bool prev_ok, float conf) const {
    const double on_th  = min_lane_conf_;
    const double off_th = min_lane_conf_ - lane_conf_hyst_;
    if (prev_ok) return conf >= off_th;
    return conf >= on_th;
  }

  nav_msgs::msg::Path fuseCenterline(const nav_msgs::msg::Path& left,
                                    const nav_msgs::msg::Path& right,
                                    bool useL,
                                    bool useR,
                                    double half_width_m,
                                    rclcpp::Time stamp) {
    nav_msgs::msg::Path center;
    center.header.frame_id = base_frame_;
    center.header.stamp = stamp;

    const bool hasL = useL && !left.poses.empty();
    const bool hasR = useR && !right.poses.empty();

    if (hasL && hasR) {
      size_t N = std::min(left.poses.size(), right.poses.size());
      center.poses.reserve(N);
      for (size_t i = 0; i < N; ++i) {
        geometry_msgs::msg::PoseStamped p;
        p.header = center.header;
        p.pose.position.x = 0.5 * (left.poses[i].pose.position.x + right.poses[i].pose.position.x);
        p.pose.position.y = 0.5 * (left.poses[i].pose.position.y + right.poses[i].pose.position.y);
        p.pose.position.z = 0.0;
        p.pose.orientation.w = 1.0;
        center.poses.push_back(p);
      }
      return center;
    }

    if (hasL && !hasR) {
      center.poses.reserve(left.poses.size());
      for (const auto& lp : left.poses) {
        geometry_msgs::msg::PoseStamped p;
        p.header = center.header;
        p.pose.position.x = lp.pose.position.x;
        p.pose.position.y = lp.pose.position.y - half_width_m;
        p.pose.position.z = 0.0;
        p.pose.orientation.w = 1.0;
        center.poses.push_back(p);
      }
      return center;
    }

    if (!hasL && hasR) {
      center.poses.reserve(right.poses.size());
      for (const auto& rp : right.poses) {
        geometry_msgs::msg::PoseStamped p;
        p.header = center.header;
        p.pose.position.x = rp.pose.position.x;
        p.pose.position.y = rp.pose.position.y + half_width_m;
        p.pose.position.z = 0.0;
        p.pose.orientation.w = 1.0;
        center.poses.push_back(p);
      }
      return center;
    }

    return center;
  }

  void leftCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto r = processOneBoundary(msg, true);
    //const std::string cam_frame = msg->header.frame_id; // best: use actual image frame
    auto left_path = boundaryToPathProjected(r, K_left_, left_cam_frame_);
    left_pub_->publish(left_path);

    if (!r.debug_bgr.empty()) {
      dbg_left_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", r.debug_bgr).toImageMsg());
    }

    {
      std::lock_guard<std::mutex> lk(m_);
      last_left_ = std::move(r);
      last_left_path_ = std::move(left_path);
    }
    tryFuseAndPublish();
  }

  void rightCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto r = processOneBoundary(msg, false);
    //const std::string cam_frame = msg->header.frame_id;
    auto right_path = boundaryToPathProjected(r, K_right_, right_cam_frame_);
    right_pub_->publish(right_path);

    if (!r.debug_bgr.empty()) {
      dbg_right_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", r.debug_bgr).toImageMsg());
    }

    {
      std::lock_guard<std::mutex> lk(m_);
      last_right_ = std::move(r);
      last_right_path_ = std::move(right_path);
    }
    tryFuseAndPublish();
  }

  // Preference update: only changes “which single-side to trust” after sustained evidence.
  void updatePreferredSide(bool left_ok, bool right_ok, float cL, float cR) {
    if (left_ok && right_ok) { prefer_switch_count_ = 0; return; }
    if (!left_ok && !right_ok) { prefer_switch_count_ = 0; return; }

    PreferredSide observed = left_ok ? PreferredSide::LEFT : PreferredSide::RIGHT;

    // Require meaningful confidence to switch.
    const float switch_min_conf = 0.45f;
    if (observed == PreferredSide::LEFT && cL < switch_min_conf) return;
    if (observed == PreferredSide::RIGHT && cR < switch_min_conf) return;

    if (observed == preferred_side_) { prefer_switch_count_ = 0; return; }

    prefer_switch_count_++;
    if (prefer_switch_count_ >= prefer_switch_threshold_) {
      preferred_side_ = observed;
      prefer_switch_count_ = 0;
    }
  }

  void publishHoldOrEmpty(rclcpp::Time stamp, float conf_val) {
    const double dt = has_last_center_pub_ ? (this->now() - last_center_pub_time_).seconds() : 1e9;
    if (has_last_center_pub_ && dt <= center_hold_sec_) {
      auto held = last_center_pub_;
      held.header.stamp = stamp;
      center_pub_->publish(held);
      std_msgs::msg::Float32 conf; conf.data = conf_val;
      conf_pub_->publish(conf);
      return;
    }
    nav_msgs::msg::Path empty;
    empty.header.frame_id = base_frame_;
    empty.header.stamp = stamp;
    center_pub_->publish(empty);
    std_msgs::msg::Float32 conf; conf.data = 0.0f;
    conf_pub_->publish(conf);
  }

  void tryFuseAndPublish() {
    std::optional<BoundaryResult> L, R;
    nav_msgs::msg::Path Lp, Rp;

    {
      std::lock_guard<std::mutex> lk(m_);
      if (!last_left_ && !last_right_) return;

      if (last_left_) {
        L = last_left_;
        Lp = last_left_path_.value_or(nav_msgs::msg::Path{});
      }
      if (last_right_) {
        R = last_right_;
        Rp = last_right_path_.value_or(nav_msgs::msg::Path{});
      }
    }

    // Choose stamp
    rclcpp::Time stamp;
    if (L && R) stamp = (L->stamp > R->stamp) ? L->stamp : R->stamp;
    else if (L) stamp = L->stamp;
    else stamp = R->stamp;

    // Freshness gating: ignore stale boundaries (critical at higher speed)
    const double now_s = this->now().seconds();
    if (L && (now_s - L->stamp.seconds()) > max_boundary_age_sec_) L.reset();
    if (R && (now_s - R->stamp.seconds()) > max_boundary_age_sec_) R.reset();

    if (!L && !R) {
      publishHoldOrEmpty(stamp, 0.05f);
      return;
    }

    // Optional sync gate (legacy) + tighter skew check when both exist
    if (L && R) {
      const double dtLR = std::abs((L->stamp - R->stamp).seconds());
      if (use_sync_gate_ && dtLR > sync_slop_sec_) return;
      if (dtLR > max_pair_skew_sec_) {
        // Prefer the newer one; drop the older to avoid mixing mismatched geometry
        if (L->stamp > R->stamp) R.reset();
        else L.reset();
      }
    }

    const float cL = L ? std::max(0.0f, L->confidence) : 0.0f;
    const float cR = R ? std::max(0.0f, R->confidence) : 0.0f;

    left_ok_  = L ? updateUsable(left_ok_,  cL) : false;
    right_ok_ = R ? updateUsable(right_ok_, cR) : false;

    updatePreferredSide(left_ok_, right_ok_, cL, cR);

    // If neither usable, hold or empty
    if (!left_ok_ && !right_ok_) {
      publishHoldOrEmpty(stamp, 0.05f);
      return;
    }

    // Auto-learn nominal half-width from good dual-lane frames
    if (auto_learn_half_width_ && left_ok_ && right_ok_) {
      double hw_med = estimateHalfWidthMedian(Lp, Rp);
      const double a = clampd(half_width_learn_alpha_, 0.0, 0.99);
      nominal_half_width_ = a * nominal_half_width_ + (1.0 - a) * hw_med;
      nominal_half_width_ = clampd(nominal_half_width_, half_width_min_m_, half_width_max_m_);
    }

    // IMPORTANT FIX for higher speed stability:
    // Do NOT “gate out” the only usable lane based on preference.
    // Preference is only used when BOTH lanes are usable but one is intermittently flaky.
    bool useL = left_ok_;
    bool useR = right_ok_;

    if (useL && useR) {
      // If both are usable, but one is much weaker, optionally follow preferred side.
      // This reduces rapid L/R toggling on curves/occlusions.
      const float dominance = 0.12f;
      if (cL + dominance < cR && preferred_side_ == PreferredSide::RIGHT) useL = false;
      if (cR + dominance < cL && preferred_side_ == PreferredSide::LEFT)  useR = false;
    }

    // Safety: if we disabled both, fall back to using the stronger lane.
    if (!useL && !useR) {
      if (left_ok_ && !right_ok_) useL = true;
      else if (!left_ok_ && right_ok_) useR = true;
      else {
        if (cL >= cR) useL = true;
        else useR = true;
      }
    }

    auto center_raw = fuseCenterline(Lp, Rp, useL, useR, nominal_half_width_, stamp);
    if (center_raw.poses.empty()) {
      publishHoldOrEmpty(stamp, 0.05f);
      return;
    }

    auto center = smoothCenterline(center_raw);

    center_pub_->publish(center);
    last_center_pub_ = center;
    last_center_pub_time_ = this->now();
    has_last_center_pub_ = !center.poses.empty();

    std_msgs::msg::Float32 conf;
    if (useL && useR)      conf.data = std::sqrt(cL * cR);
    else if (useL)         conf.data = cL;
    else                   conf.data = cR;
    conf_pub_->publish(conf);

    if (debug_print_period_ms_ > 0) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), (uint64_t)debug_print_period_ms_,
        "conf=%.3f (L=%.3f %s, R=%.3f %s) halfW=%.3f roi_top_frac=%.2f ageL=%.2f ageR=%.2f",
        conf.data,
        cL, useL ? "OK" : "NO",
        cR, useR ? "OK" : "NO",
        nominal_half_width_,
        roi_top_frac_,
        L ? (float)(this->now() - L->stamp).seconds() : 99.0f,
        R ? (float)(this->now() - R->stamp).seconds() : 99.0f
      );
    }
  }

  // -----------------------------
  // Params
  // -----------------------------


  std::string left_topic_, right_topic_, frame_;
  double roi_top_frac_{0.35};
  //double mppx_{0.025}, mppy_{0.025};
  double nominal_half_width_{1.50};
  //double left_cam_y_off_{0.30}, right_cam_y_off_{-0.30};

  int min_points_fit_{15};
  int support_pts_lo_{150};
  int support_pts_hi_{900};
  double residual_good_px_{6.0};
  double residual_bad_px_{18.0};
  double curv_good_{1e-4};
  double curv_bad_{8e-4};

  double ground_z_in_base_{0.0};


  double debug_print_hz_{1.0};
  int debug_print_period_ms_{1000};

  // ROS
  image_transport::Subscriber left_sub_, right_sub_;
  image_transport::Publisher dbg_left_pub_, dbg_right_pub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr center_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr right_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr conf_pub_;

  // State
  std::mutex m_;
  std::optional<BoundaryResult> last_left_;
  std::optional<BoundaryResult> last_right_;
  std::optional<nav_msgs::msg::Path> last_left_path_;
  std::optional<nav_msgs::msg::Path> last_right_path_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneDetectorDualNode>());
  rclcpp::shutdown();
  return 0;
}