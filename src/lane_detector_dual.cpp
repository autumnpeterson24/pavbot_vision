#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/msg/Laser_scan.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <opencv2/opencv.hpp>
#include <mutex>
#include <optional>

struct Poly2 {
  double a{0}, b{0}, c{0};
  bool valid{false};
  double eval(double y) const { return a*y*y + b*y + c; }
};

static Poly2 fit_poly2(const std::vector<cv::Point>& pts) {
  Poly2 poly;
  if (pts.size() < 15) return poly;

  cv::Mat A(pts.size(), 3, CV_64F), X(pts.size(), 1, CV_64F);
  for (size_t i = 0; i < pts.size(); ++i) {
    double y = pts[i].y, x = pts[i].x;
    A.at<double>(i,0) = y*y;
    A.at<double>(i,1) = y;
    A.at<double>(i,2) = 1.0;
    X.at<double>(i,0) = x;
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
    int x = (int)std::round(p.eval(y));
    if (0 <= x && x < img.cols) cv::circle(img, {x, y}, 2, col, -1);
  }
}

struct BoundaryResult {
  Poly2 poly;
  float confidence{0.f};
  cv::Mat debug_bgr;          // ROI-sized debug
  int roi_top{0};
  int W{0}, H{0};
  rclcpp::Time stamp;
};

class LaneDetectorDualNode : public rclcpp::Node {
public:
  LaneDetectorDualNode() : Node("lane_detector_dual") {
    declare_parameter<std::string>("left_camera_topic",  "/left_cam/image_raw");
    declare_parameter<std::string>("right_camera_topic", "/right_cam/image_raw");
    declare_parameter<std::string>("path_frame", "base_link");

    declare_parameter<double>("roi_top_frac", 0.35);
    declare_parameter<double>("mppx", 0.025);
    declare_parameter<double>("mppy", 0.025);

    // If only one boundary is visible, we “invent” the other side using this.
    declare_parameter<double>("nominal_lane_half_width_m", 0.75);

    // If cameras are physically offset from base_link, you can compensate roughly here.
    // Positive y is left, negative y is right (ROS standard).
    declare_parameter<double>("left_cam_y_offset_m",  0.30);
    declare_parameter<double>("right_cam_y_offset_m", -0.30);

    // Sync window for the two images
    declare_parameter<double>("sync_slop_sec", 0.08);

    get_parameter("left_camera_topic", left_topic_);
    get_parameter("right_camera_topic", right_topic_);
    get_parameter("path_frame", frame_);
    get_parameter("roi_top_frac", roi_top_frac_);
    get_parameter("mppx", mppx_);
    get_parameter("mppy", mppy_);
    get_parameter("nominal_lane_half_width_m", nominal_half_width_);
    get_parameter("left_cam_y_offset_m", left_cam_y_off_);
    get_parameter("right_cam_y_offset_m", right_cam_y_off_);
    get_parameter("sync_slop_sec", sync_slop_sec_);

    center_pub_ = create_publisher<nav_msgs::msg::Path>("/lanes/centerline", 10);
    left_pub_   = create_publisher<nav_msgs::msg::Path>("/lanes/left_boundary", 10);
    right_pub_  = create_publisher<nav_msgs::msg::Path>("/lanes/right_boundary", 10);
    conf_pub_   = create_publisher<std_msgs::msg::Float32>("/lanes/confidence", 10);

    dbg_left_pub_  = image_transport::create_publisher(this, "/lanes/debug_left");
    dbg_right_pub_ = image_transport::create_publisher(this, "/lanes/debug_right");

    left_sub_ = image_transport::create_subscription(
      this, left_topic_,
      std::bind(&LaneDetectorDualNode::leftCb, this, std::placeholders::_1),
      "raw");

    right_sub_ = image_transport::create_subscription(
      this, right_topic_,
      std::bind(&LaneDetectorDualNode::rightCb, this, std::placeholders::_1),
      "raw");

    RCLCPP_INFO(get_logger(), "Dual lane detector subscribed to:");
    RCLCPP_INFO(get_logger(), "  left : %s", left_topic_.c_str());
    RCLCPP_INFO(get_logger(), "  right: %s", right_topic_.c_str());
  }

private:
  // --- Core per-camera processing: produce ONE boundary curve (poly) ---
  BoundaryResult processOneBoundary(const sensor_msgs::msg::Image::ConstSharedPtr& msg,
                                   bool is_left_camera) {
    BoundaryResult out;
    out.stamp = msg->header.stamp;

    cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    out.W = bgr.cols;
    out.roi_top = (int)(bgr.rows * roi_top_frac_);
    cv::Mat roi = bgr.rowRange(out.roi_top, bgr.rows);
    out.H = roi.rows;

    // Threshold (white-ish + edges), same as your existing approach
    cv::Mat hsv; cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch; cv::split(hsv, ch);
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
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  cv::getStructuringElement(cv::MORPH_RECT, {3,3}));

    // We expect ONE lane boundary per camera, so focus histogram search to the “expected side”.
    const int H = mask.rows, W = mask.cols;
    cv::Mat hist;
    cv::reduce(mask.rowRange(H/2, H), hist, 0, cv::REDUCE_SUM, CV_32S);

    int x_base = (is_left_camera ? W/3 : (2*W)/3);
    int x_lo = is_left_camera ? 0 : W/2;
    int x_hi = is_left_camera ? W/2 : W;

    double best = 0;
    for (int x = x_lo; x < x_hi; ++x) {
      int v = hist.at<int>(0, x);
      if (v > best) { best = (double)v; x_base = x; }
    }

    // Sliding windows but only for one line
    const int nwindows = 9;
    const int winH = H / nwindows;
    int margin = 60, minpix = 50;

    std::vector<cv::Point> P;
    int x_curr = x_base;
    for (int w = 0; w < nwindows; ++w) {
      int ylo = H - (w + 1) * winH;
      int yhi = H - w * winH;

      int xlo = std::max(0, x_curr - margin);
      int xhi2 = std::min(W - 1, x_curr + margin);

      std::vector<cv::Point> nz;
      cv::findNonZero(mask(cv::Rect(xlo, ylo, xhi2 - xlo + 1, yhi - ylo)), nz);

      for (auto &p : nz) P.emplace_back(p.x + xlo, p.y + ylo);

      if ((int)nz.size() > minpix) {
        int s = 0;
        for (auto &p : nz) s += p.x + xlo;
        x_curr = s / (int)nz.size();
      }
    }

    out.poly = fit_poly2(P);

    // Confidence: mask density
    out.confidence = mask.empty() ? 0.f : (float)cv::countNonZero(mask) / (float)mask.total();

    // Debug
    cv::Mat dbg; cv::cvtColor(mask, dbg, cv::COLOR_GRAY2BGR);
    if (out.poly.valid) {
      draw_poly(dbg, out.poly, cv::Scalar(0,255,0));
    }
    out.debug_bgr = dbg;
    return out;
  }

  // Convert a polynomial boundary (in ROI image coordinates) to a Path in base_link.
  nav_msgs::msg::Path boundaryToPath(const BoundaryResult& r,
                                     double cam_y_offset_m) {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_;
    path.header.stamp = r.stamp;

    if (!r.poly.valid) return path;

    const double cx_img = r.W / 2.0;
    for (int y = r.H - 1; y >= 0; y -= 5) {
      double x_img = r.poly.eval(y);
      geometry_msgs::msg::PoseStamped p;
      p.header = path.header;

      // Your existing mapping (temporary). Replace later with IPM/homography.
      p.pose.position.x = (r.H - 1 - y) * mppy_;                 // forward
      p.pose.position.y = (x_img - cx_img) * mppx_ + cam_y_offset_m; // lateral + camera offset
      p.pose.position.z = 0.0;
      p.pose.orientation.w = 1.0;
      path.poses.push_back(p);
    }
    return path;
  }

  // Build centerline from left/right boundary paths (pointwise average by index).
  nav_msgs::msg::Path fuseCenterline(const nav_msgs::msg::Path& left,
                                    const nav_msgs::msg::Path& right,
                                    rclcpp::Time stamp) {
    nav_msgs::msg::Path center;
    center.header.frame_id = frame_;
    center.header.stamp = stamp;

    const bool hasL = !left.poses.empty();
    const bool hasR = !right.poses.empty();

    // If both exist, average them (assumes same sampling).
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

    // If only one side exists, offset inward by nominal lane half width.
    if (hasL && !hasR) {
      center.poses.reserve(left.poses.size());
      for (const auto& lp : left.poses) {
        geometry_msgs::msg::PoseStamped p;
        p.header = center.header;
        p.pose.position.x = lp.pose.position.x;
        p.pose.position.y = lp.pose.position.y - nominal_half_width_; // move rightward
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
        p.pose.position.y = rp.pose.position.y + nominal_half_width_; // move leftward
        p.pose.position.z = 0.0;
        p.pose.orientation.w = 1.0;
        center.poses.push_back(p);
      }
      return center;
    }

    return center; // empty
  }

 void leftCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  auto r = processOneBoundary(msg, true);

  // Publish left boundary/debug immediately
  auto left_path = boundaryToPath(r, left_cam_y_off_);
  left_pub_->publish(left_path);

  if (!r.debug_bgr.empty()) {
    dbg_left_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", r.debug_bgr).toImageMsg());
  }

  {
    std::lock_guard<std::mutex> lk(m_);
    last_left_ = std::move(r);
  }
  tryFuseAndPublish();  // only for centerline
}


void rightCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
  auto r = processOneBoundary(msg, false);

  // Publish right boundary/debug immediately
  auto right_path = boundaryToPath(r, right_cam_y_off_);
  right_pub_->publish(right_path);

  if (!r.debug_bgr.empty()) {
    dbg_right_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", r.debug_bgr).toImageMsg());
  }

  {
    std::lock_guard<std::mutex> lk(m_);
    last_right_ = std::move(r);
  }
  tryFuseAndPublish();  // only for centerline
}


  void tryFuseAndPublish() {
    std::optional<BoundaryResult> L, R;
    {
      std::lock_guard<std::mutex> lk(m_);
      L = last_left_;
      R = last_right_;
    }
    if (!L || !R) return;

    // // Timestamp sync gate
    // const double dt = std::abs((L->stamp - R->stamp).seconds());
    // if (dt > sync_slop_sec_) return;

    // Publish boundary paths
    auto left_path  = boundaryToPath(*L, left_cam_y_off_);
    auto right_path = boundaryToPath(*R, right_cam_y_off_);

    left_pub_->publish(left_path);
    right_pub_->publish(right_path);

    // Centerline
    auto stamp = (L->stamp > R->stamp) ? L->stamp : R->stamp;
    auto center = fuseCenterline(left_path, right_path, stamp);
    center_pub_->publish(center);

    // Confidence: combine
    std_msgs::msg::Float32 conf;
    conf.data = 0.5f * (L->confidence + R->confidence);
    conf_pub_->publish(conf);

    // Debug images
    if (!L->debug_bgr.empty()) {
      dbg_left_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", L->debug_bgr).toImageMsg());
    }
    if (!R->debug_bgr.empty()) {
      dbg_right_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", R->debug_bgr).toImageMsg());
    }
  }

private:
  std::string left_topic_, right_topic_, frame_;
  double roi_top_frac_{0.35};
  double mppx_{0.025}, mppy_{0.025};
  double nominal_half_width_{0.75};
  double left_cam_y_off_{0.30}, right_cam_y_off_{-0.30};
  double sync_slop_sec_{0.08};

  image_transport::Subscriber left_sub_, right_sub_;
  image_transport::Publisher dbg_left_pub_, dbg_right_pub_;

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr center_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr right_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr conf_pub_;

  std::mutex m_;
  std::optional<BoundaryResult> last_left_;
  std::optional<BoundaryResult> last_right_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneDetectorDualNode>());
  rclcpp::shutdown();
  return 0;
}
