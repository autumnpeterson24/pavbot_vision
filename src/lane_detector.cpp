#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/opencv.hpp>

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
    if (x >= 0 && x < img.cols) {
      cv::circle(img, {x, y}, 2, col, -1);
    }
  }
}

class LaneDetectorNode : public rclcpp::Node {
public:
  LaneDetectorNode() : Node("lane_detector") {
    declare_parameter<std::string>("camera_topic", "/camera/image_raw");
    declare_parameter<std::string>("path_frame", "base_link");
    declare_parameter<double>("roi_top_frac", 0.35);
    declare_parameter<double>("mppx", 0.025);
    declare_parameter<double>("mppy", 0.025);

    get_parameter("camera_topic", cam_topic_);
    get_parameter("path_frame", frame_);
    get_parameter("roi_top_frac", roi_top_frac_);
    get_parameter("mppx", mppx_);
    get_parameter("mppy", mppy_);

    path_pub_ = create_publisher<nav_msgs::msg::Path>("/lanes/centerline", 10);
    conf_pub_ = create_publisher<std_msgs::msg::Float32>("/lanes/confidence", 10);
    dbg_pub_  = image_transport::create_publisher(this, "/lanes/debug_image");

    sub_ = image_transport::create_subscription(
      this, cam_topic_,
      std::bind(&LaneDetectorNode::cb, this, std::placeholders::_1),
      "raw");

    RCLCPP_INFO(get_logger(), "LaneDetector subscribed to %s", cam_topic_.c_str());
  }

private:
  void cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    // 1) Grab frame & ROI
    cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    const int top = (int)(frame.rows * roi_top_frac_);
    cv::Mat roi = frame.rowRange(top, frame.rows);

    // 2) Threshold (white-ish + edges)
    cv::Mat hsv; cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch; cv::split(hsv, ch);
    cv::Mat maskS, maskV, mask;
    cv::threshold(ch[1], maskS, 60, 255, cv::THRESH_BINARY_INV);  // low saturation
    cv::threshold(ch[2], maskV, 200, 255, cv::THRESH_BINARY);     // bright
    cv::bitwise_and(maskS, maskV, mask);

    cv::Mat gray, gradx, absx, edges;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, gradx, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(gradx, absx);
    cv::threshold(absx, edges, 40, 255, cv::THRESH_BINARY);

    cv::bitwise_or(mask, edges, mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  cv::getStructuringElement(cv::MORPH_RECT, {3,3}));

    // 3) Sliding windows to collect lane pixels
    const int H = mask.rows, W = mask.cols, mid = W / 2;
    cv::Mat hist; cv::reduce(mask.rowRange(H/2, H), hist, 0, cv::REDUCE_SUM, CV_32S);

    int leftx = 0, rightx = 0; double ml = 0, mr = 0;
    for (int x = 0; x < mid; ++x)  { int v = hist.at<int>(0,x); if (v > ml) { ml = v; leftx = x; } }
    for (int x = mid; x < W; ++x) { int v = hist.at<int>(0,x); if (v > mr) { mr = v; rightx = x; } }

    const int nwindows = 9;
    const int winH = H / nwindows;
    int margin = 60, minpix = 50;

    std::vector<cv::Point> L, R;
    for (int w = 0; w < nwindows; ++w) {
      int ylo = H - (w + 1) * winH;
      int yhi = H - w * winH;

      int lxlo = std::max(0, leftx  - margin), lxhi = std::min(W - 1, leftx  + margin);
      int rxlo = std::max(0, rightx - margin), rxhi = std::min(W - 1, rightx + margin);

      std::vector<cv::Point> nl, nr;
      cv::findNonZero(mask(cv::Rect(lxlo, ylo, lxhi - lxlo + 1, yhi - ylo)), nl);
      cv::findNonZero(mask(cv::Rect(rxlo, ylo, rxhi - rxlo + 1, yhi - ylo)), nr);

      for (auto &p : nl) L.emplace_back(p.x + lxlo, p.y + ylo);
      for (auto &p : nr) R.emplace_back(p.x + rxlo, p.y + ylo);

      if ((int)nl.size() > minpix) {
        int s = 0; for (auto &p : nl) s += p.x + lxlo;
        leftx = s / (int)nl.size();
      }
      if ((int)nr.size() > minpix) {
        int s = 0; for (auto &p : nr) s += p.x + rxlo;
        rightx = s / (int)nr.size();
      }
    }

    // 4) Fit quadratics x(y) for left/right
    Poly2 pl = fit_poly2(L), pr = fit_poly2(R);

    // 5) Build centerline Path in meters (base_link frame)
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_;
    path.header.stamp = msg->header.stamp;

    const double cx_img = W / 2.0;
    for (int y = H - 1; y >= 0; y -= 5) {
      double x;
      if (pl.valid && pr.valid)      x = 0.5 * (pl.eval(y) + pr.eval(y));
      else if (pl.valid)             x = pl.eval(y);
      else if (pr.valid)             x = pr.eval(y);
      else                           continue;

      geometry_msgs::msg::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = (H - 1 - y) * mppy_;      // forward (meters)
      p.pose.position.y = (x - cx_img) * mppx_;     // lateral (meters)
      p.pose.position.z = 0.0;
      p.pose.orientation.w = 1.0;
      path.poses.push_back(p);
    }
    path_pub_->publish(path);

    // 6) Confidence (fraction of mask pixels ON)
    std_msgs::msg::Float32 conf;
    conf.data = mask.empty() ? 0.f : (float)cv::countNonZero(mask) / (float)mask.total();
    conf_pub_->publish(conf);

    // 7) Debug image: draw red centerline + green/blue boundaries
    cv::Mat dbg; cv::cvtColor(mask, dbg, cv::COLOR_GRAY2BGR);

    // Red centerline dots
    for (int y = H - 1; y >= 0; y -= 5) {
      double x;
      if (pl.valid && pr.valid)      x = 0.5 * (pl.eval(y) + pr.eval(y));
      else if (pl.valid)             x = pl.eval(y);
      else if (pr.valid)             x = pr.eval(y);
      else                           continue;

      int xi = (int)std::round(x);
      if (0 <= xi && xi < W) {
        cv::circle(dbg, cv::Point(xi, y), 2, cv::Scalar(0, 0, 255), -1); // red
      }
    }

    if (pl.valid) draw_poly(dbg, pl, cv::Scalar(0,255,0));   // green
    if (pr.valid) draw_poly(dbg, pr, cv::Scalar(255,0,0));   // blue

    dbg_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", dbg).toImageMsg());
  }

  // --- Members ---
  std::string cam_topic_, frame_;
  double roi_top_frac_, mppx_, mppy_;
  image_transport::Subscriber sub_;
  image_transport::Publisher  dbg_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr conf_pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
