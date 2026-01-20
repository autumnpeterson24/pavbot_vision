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

struct Poly2 {
  double a{0}, b{0}, c{0};
  bool valid{false};
  double eval(double y) const { return a*y*y + b*y + c; }
};

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

// ---- confidence helpers ----
static inline float clamp01(float v) {
  return std::max(0.0f, std::min(1.0f, v));
}

static inline float smoothstep(float edge0, float edge1, float x) {
  if (edge1 <= edge0) return (x >= edge1) ? 1.0f : 0.0f;
  float t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3.0f - 2.0f * t);
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
    // Topics / frames
    declare_parameter<std::string>("left_camera_topic",  "/left_cam/image_raw");
    declare_parameter<std::string>("right_camera_topic", "/right_cam/image_raw");
    declare_parameter<std::string>("path_frame", "base_link");

    // Geometry conversion
    declare_parameter<double>("roi_top_frac", 0.35);
    declare_parameter<double>("mppx", 0.025);
    declare_parameter<double>("mppy", 0.025);

    // If only one boundary exists
    declare_parameter<double>("nominal_lane_half_width_m", 0.75);

    // Camera offsets relative to base_link (ROS: +y left, -y right)
    declare_parameter<double>("left_cam_y_offset_m",  0.30);
    declare_parameter<double>("right_cam_y_offset_m", -0.30);

    // Optional timestamp sync gate (disabled by default)
    declare_parameter<bool>("use_sync_gate", false);
    declare_parameter<double>("sync_slop_sec", 0.08);

    // Confidence tuning (defaults are reasonable for 640x480 sim)
    declare_parameter<int>("min_points_fit", 15);
    declare_parameter<int>("support_pts_lo", 150);
    declare_parameter<int>("support_pts_hi", 900);
    declare_parameter<double>("residual_good_px", 6.0);
    declare_parameter<double>("residual_bad_px", 18.0);
    declare_parameter<double>("curv_good", 1e-4);
    declare_parameter<double>("curv_bad", 8e-4);

    // Debug throttling
    declare_parameter<double>("debug_print_hz", 1.0);

    // Load params
    get_parameter("left_camera_topic", left_topic_);
    get_parameter("right_camera_topic", right_topic_);
    get_parameter("path_frame", frame_);

    get_parameter("roi_top_frac", roi_top_frac_);
    get_parameter("mppx", mppx_);
    get_parameter("mppy", mppy_);

    get_parameter("nominal_lane_half_width_m", nominal_half_width_);
    get_parameter("left_cam_y_offset_m", left_cam_y_off_);
    get_parameter("right_cam_y_offset_m", right_cam_y_off_);

    get_parameter("use_sync_gate", use_sync_gate_);
    get_parameter("sync_slop_sec", sync_slop_sec_);

    get_parameter("min_points_fit", min_points_fit_);
    get_parameter("support_pts_lo", support_pts_lo_);
    get_parameter("support_pts_hi", support_pts_hi_);
    get_parameter("residual_good_px", residual_good_px_);
    get_parameter("residual_bad_px", residual_bad_px_);
    get_parameter("curv_good", curv_good_);
    get_parameter("curv_bad", curv_bad_);

    get_parameter("debug_print_hz", debug_print_hz_);
    debug_print_period_ms_ = (debug_print_hz_ <= 0.0) ? 0 : (int)(1000.0 / debug_print_hz_);

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

    RCLCPP_INFO(get_logger(), "lane_detector_dual up.");
    RCLCPP_INFO(get_logger(), "  left : %s", left_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  right: %s", right_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  frame: %s", frame_.c_str());
    RCLCPP_INFO(get_logger(), "  roi_top_frac=%.2f  mppx=%.4f  mppy=%.4f", roi_top_frac_, mppx_, mppy_);
  }

private:
  BoundaryResult processOneBoundary(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                                   bool is_left_camera) {
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
    cv::threshold(ch[1], maskS, 60, 255, cv::THRESH_BINARY_INV);
    cv::threshold(ch[2], maskV, 200, 255, cv::THRESH_BINARY);
    cv::bitwise_and(maskS, maskV, mask);

    cv::Mat gray, gradx, absx, edges;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, gradx, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(gradx, absx);
    cv::threshold(absx, edges, 40, 255, cv::THRESH_BINARY);

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

      // accumulate points in ROI coords
      for (auto &p : nz) P.emplace_back(p.x + xlo2, p.y + ylo);

      // recenter if enough
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

    out.poly = fit_poly2(P);

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

      // C) Fit residual
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

      // D) Side sanity
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

    // Debug image (ROI-sized)
    cv::Mat dbg;
    cv::cvtColor(mask, dbg, cv::COLOR_GRAY2BGR);
    if (out.poly.valid) draw_poly(dbg, out.poly, cv::Scalar(0,255,0));
    out.debug_bgr = dbg;

    return out;
  }

  nav_msgs::msg::Path boundaryToPath(const BoundaryResult& r, double cam_y_offset_m) {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_;
    path.header.stamp = r.stamp;

    if (!r.poly.valid || r.W <= 0 || r.H <= 0) return path;

    const double cx_img = 0.5 * (double)r.W;

    for (int y = r.H - 1; y >= 0; y -= 5) {
      double x_img = r.poly.eval((double)y);
      if (!std::isfinite(x_img)) continue;

      x_img = std::min(std::max(x_img, 0.0), (double)(r.W - 1));

      geometry_msgs::msg::PoseStamped p;
      p.header = path.header;

      const double forward_px = (double)((r.H - 1) - y);
      const double lateral_px = x_img - cx_img;

      // forward/lateral mapping
      p.pose.position.x = forward_px * mppx_;
      p.pose.position.y = lateral_px * mppy_ + cam_y_offset_m;
      p.pose.position.z = 0.0;
      p.pose.orientation.w = 1.0;

      path.poses.push_back(p);
    }

    return path;
  }

  nav_msgs::msg::Path fuseCenterline(const nav_msgs::msg::Path& left,
                                     const nav_msgs::msg::Path& right,
                                     rclcpp::Time stamp) {
    nav_msgs::msg::Path center;
    center.header.frame_id = frame_;
    center.header.stamp = stamp;

    const bool hasL = !left.poses.empty();
    const bool hasR = !right.poses.empty();

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
        p.pose.position.y = lp.pose.position.y - nominal_half_width_;
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
        p.pose.position.y = rp.pose.position.y + nominal_half_width_;
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

    // publish left boundary and debug
    auto left_path = boundaryToPath(r, left_cam_y_off_);
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

    // publish right boundary and debug
    auto right_path = boundaryToPath(r, right_cam_y_off_);
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

  void tryFuseAndPublish() {
    std::optional<BoundaryResult> L, R;
    nav_msgs::msg::Path Lp, Rp;

    {
      std::lock_guard<std::mutex> lk(m_);
      if (!last_left_ || !last_right_) return;
      L = last_left_;
      R = last_right_;
      Lp = last_left_path_.value_or(nav_msgs::msg::Path{});
      Rp = last_right_path_.value_or(nav_msgs::msg::Path{});
    }

    if (use_sync_gate_) {
      const double dt = std::abs((L->stamp - R->stamp).seconds());
      if (dt > sync_slop_sec_) return;
    }

    // Centerline
    const auto stamp = (L->stamp > R->stamp) ? L->stamp : R->stamp;
    auto center = fuseCenterline(Lp, Rp, stamp);
    center_pub_->publish(center);

    // Confidence combine: geometric mean (penalizes if either side is poor)
    std_msgs::msg::Float32 conf;
    const float cL = std::max(0.0f, L->confidence);
    const float cR = std::max(0.0f, R->confidence);
    conf.data = std::sqrt(cL * cR);
    conf_pub_->publish(conf);

    // Throttled debug print
    if (debug_print_period_ms_ > 0) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), (uint64_t)debug_print_period_ms_,
        "conf=%.3f (L=%.3f R=%.3f) "
        "L_pts=%d L_hit=%d/%d L_res=%.1fpx L_side=%.2f | "
        "R_pts=%d R_hit=%d/%d R_res=%.1fpx R_side=%.2f | "
        "mppx=%.4f mppy=%.4f roi_top_frac=%.2f W=%d H=%d",
        conf.data, L->confidence, R->confidence,
        L->num_points, L->windows_hit, L->nwindows, L->mean_residual_px, L->frac_in_expected_half,
        R->num_points, R->windows_hit, R->nwindows, R->mean_residual_px, R->frac_in_expected_half,
        mppx_, mppy_, roi_top_frac_, L->W, L->H
      );
    }
  }

private:
  // params
  std::string left_topic_, right_topic_, frame_;
  double roi_top_frac_{0.35};
  double mppx_{0.025}, mppy_{0.025};
  double nominal_half_width_{0.75};
  double left_cam_y_off_{0.30}, right_cam_y_off_{-0.30};

  bool use_sync_gate_{false};
  double sync_slop_sec_{0.08};

  int min_points_fit_{15};
  int support_pts_lo_{150};
  int support_pts_hi_{900};
  double residual_good_px_{6.0};
  double residual_bad_px_{18.0};
  double curv_good_{1e-4};
  double curv_bad_{8e-4};

  double debug_print_hz_{1.0};
  int debug_print_period_ms_{1000};

  // ROS
  image_transport::Subscriber left_sub_, right_sub_;
  image_transport::Publisher dbg_left_pub_, dbg_right_pub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr center_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr right_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr conf_pub_;

  // state
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
