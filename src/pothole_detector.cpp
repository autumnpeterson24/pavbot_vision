/*
  pothole_detector.cpp =============================
  Author: Autumn Peterson (IGVC 2026 PAVbot Team)
  Assistance from the following:
    Pothole Detection Using Image Processing and Machine Learning: https://ieeexplore.ieee.org/document/11020129
    A Review of Vision-Based Pothole Detection Methods Using Computer Vision and Machine Learning: https://www.mdpi.com/1424-8220/24/17/5652  
    Advancements in pothole detection techniques: a comprehensive review and comparative analysis: https://link.springer.com/article/10.1007/s44163-025-00297-7 
    Ros Packages from ROS wiki for OpenCV: https://wiki.ros.org/vision_opencv
    OpenVC libraries: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    Morphology Filters: https://www.theobjects.com/dragonfly/dfhelp/4-0/Content/05_Image%20Processing/Morphology%20Filters.htm#:~:text=In%20mathematical%20morphology%20and%20digital,closing%20and%20the%20input%20image.


 Description:
  ROS 2 (Humble) node performs real-time pothole detection using classical computer-vision techniques optimized for embedded systems and IGVC AutoNav environments. 
  Subscribes to a camera image topic, processes each frame using OpenCV, and detects dark regions on the road surface that resemble potholes.

  The algorithm supports two operation modes:
   - "blackhat": optimized for outdoor asphalt detection using morphological 
     black-hat filtering and adaptive thresholding to isolate dark depressions 
     under variable lighting.
   - "simple": simplified threshold-based mode for indoor or laboratory testing 
     (e.g., black circles on white surfaces).

  Each detected region is filtered by area and roundness, merged if overlapping, 
  and published as 2D bounding boxes in standard ROS Vision messages.

  Published Topics:
   - /potholes/detections (vision_msgs/Detection2DArray): 
       Bounding boxes of detected potholes in image coordinates.
   - /potholes/confidence (std_msgs/Float32): 
       Heuristic detection confidence based on number of blobs.
   - /potholes/debug_image (sensor_msgs/Image): 
       Visualization of the detection overlay, threshold mask, or both.

  This node is designed for real-time execution on low-power hardware 
  such as NVIDIA Jetson platforms, using only CPU-based OpenCV operations 
  (no neural inference). Its parameters are fully configurable at runtime, 
  allowing for tuning between testing and outdoor conditions.

  ** Setups for blackhat/outdoors:

      mode:=blackhat
      blackhat_kernel:=17
      adaptive_block:=31
      adaptive_c:=-7
      close_kernel:=7
      use_connected_components:=false
      merge_boxes:=true
      min_area:=300 max_area:=80000 roundness_min:=0.25

  ** Setups for whiteboard test:

      mode:=simple
      simple_bias:=25      
      close_kernel:=11
      use_connected_components:=true
      merge_boxes:=true
      min_area:=40 roundness_min:=0.15

 ======================================================
 */

#include <rclcpp/rclcpp.hpp> 
#include <image_transport/image_transport.hpp> // ROS2 ndoes for image pub/sub
#include <cv_bridge/cv_bridge.h> //for OpenCV
// ROS messages
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <opencv2/opencv.hpp>

class PotholeDetector : public rclcpp::Node {
  /* Node for Detecting Potholes and processing frame by frame */
public:
  PotholeDetector() : Node("pothole_detector") {
    // Core parameters ----------------------------
    cam_topic  = this->declare_parameter<std::string>("image_topic", "/pothole_cam/image_raw"); // subscribing to the images from the pothole_cam (created from cam_pub.cpp)
    min_area   = this->declare_parameter<int>("min_area", 200); // Threshold for rejecting tiny "specks"
    max_area  = this->declare_parameter<int>("max_area", 25000); // Thresholds for large black regions that could be shadows mistaken for potholes

    round_min = this->declare_parameter<double>("roundness_min", 0.35); // ensure that potholes are a "blobby" shape and not a crack 
    // ^ uses circularity metric (how circular something is) 
    // 4*pi*A / P^2 is an element of (0, 1]; P = perimeter, A = area

    dbg = this->declare_parameter<bool>("debug", true); // publishes the debug image to check for testing
    // ----------------------------------------------

    // Mode and threshold params --------------------
    mode    = this->declare_parameter<std::string>("mode", "blackhat"); // mode for "blackhat" (asphalt) or "simple" (whiteboard test)

    // blackhat tuning (uses the black tophat filter to isolate and extract dark areas from the image that are smaller than the overall image)
    
    // kernel size for extracting certain color elements from a photo (for us that is black for potholes) bigger the kernel = bigger the extraction
    bh_kernel = this->declare_parameter<int>("blackhat_kernel", 15); 

    // blocks for adaptive Gaussian Threshold (instead of taking in whole image take in small regions/neighboorhoods of pixels (local analysis) to get local threshold for each pixel
    // creates average of pixels around it within the neighborhoods to make a better overall thresholding for the image
    // Good for uneven lighting
    bh_block = this->declare_parameter<int>("adaptive_block", 31);
    bh_c = this->declare_parameter<int>("adaptive_c", -5); // Constant that is subtracted to fine tune/ bias towards darker regions

    // simple tuning for whiteboard testing (Uses Otsu's Threshold instead so very black regions become the foreground)
    simple_bias   = this->declare_parameter<int>("simple_bias", 15); // lower = stricter consideration
    simple_merge_px = this->declare_parameter<int>("simple_merge_px", 5); // fuse pixels here to dilate -> erode and fuse the speckles
    close_k    = this->declare_parameter<int>("close_kernel", 7); // sealing the gaps within the kernel
    // ----------------------------------------------------------------

    // Post-processing options ----------------------------------------
    use_cc   = this->declare_parameter<bool>("use_connected_components", false); // connected components = one blob per box
    merge_boxes = this->declare_parameter<bool>("merge_boxes", true); // merges overlapping or nearby rectangles (lots of fragmenting can happen with blobs i.e. tons of small squares instead of overall big bounding box)
    pad_px   = this->declare_parameter<int>("pad_px", 6); // pads pounding boxes to contain the pothole within it
    iou_merge = this->declare_parameter<double>("iou_merge", 0.20); // threshold for Intersection Over Union for deciding to merge
    debug_view = this->declare_parameter<std::string>("debug_view", "both"); // for debug view in Rviz
    // -----------------------------------------------------------------

    // ROS pub/sub -----------------------------------------------------
    // Publishing the detections found
    detections_pub = this->create_publisher<vision_msgs::msg::Detection2DArray>("/potholes/detections", 10);

    // Publishing the confidence of the detections (good for seeing if node is working)
    conf_pub  = this->create_publisher<std_msgs::msg::Float32>("/potholes/confidence", 10);
    dbg_pub   = image_transport::create_publisher(this, "/potholes/debug_image");

    // subscription for the pothole_cam
    // simple subscriber right now. Add more robust when eventually digesting image and creating poses for robot to go to so it can avoid pothole
    sub = image_transport::create_subscription( this, cam_topic, std::bind(&PotholeDetector::imageCb, this, std::placeholders::_1), "raw", rmw_qos_profile_sensor_data); 

    RCLCPP_INFO(get_logger(), "PotholeDetector subscribed to %s (mode=%s, bias=%d)", cam_topic.c_str(), mode.c_str(), simple_bias); // log to see if ndoe properly subscribed
  }

private:
  // Helpers ---------------------------------------------------------------
  static cv::Rect padRect(const cv::Rect& r, int p, const cv::Size& limit) {
    /* Pads rectangles out by a certain amount when creating boundary boxes */
    cv::Rect out = r;
    out.x = std::max(0, out.x - p);
    out.y = std::max(0, out.y - p);
    out.width  = std::min(limit.width  - out.x,  out.width  + 2*p);
    out.height = std::min(limit.height - out.y,  out.height + 2*p);
    return out;
  }

  static double IoU(const cv::Rect& a, const cv::Rect& b) {
    /* Helper funtion to make sure tiny gaps that are close to one another don't produce a bunch of boxes */
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width,  b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int iw = std::max(0, x2 - x1), ih = std::max(0, y2 - y1);
    int inter = iw * ih;
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? (double)inter / (double)uni : 0.0;
  }

  static double roundnessFromContour(const std::vector<cv::Point>& c) {
    /* Checking the roundess of a pothole to make sure a crack is not considered */
    double A = std::max(1.0, cv::contourArea(c));
    double P = std::max(1e-6, cv::arcLength(c, true));
    return 4.0 * M_PI * A / (P * P);
  }

  void mergeNearbyBoxes(std::vector<cv::Rect>& boxes, const cv::Size& imgSize) {
    /* Greedy union of nearby boxes to merge them all into a big bounding box */
    if (boxes.empty()) return;
    std::vector<cv::Rect> merged;
    std::vector<bool> used(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
      if (used[i]) continue;
      cv::Rect acc = padRect(boxes[i], pad_px, imgSize);
      used[i] = true;
      bool grew = true;
      while (grew) {
        grew = false;
        for (size_t j = 0; j < boxes.size(); ++j) {
          if (used[j]) continue;
          cv::Rect pj = padRect(boxes[j], pad_px, imgSize);
          if (IoU(acc, pj) > iou_merge || (acc & pj).area() > 0) {
            acc |= pj;
            used[j] = true;
            grew = true;
          }
        }
      }
      merged.push_back(acc);
    }
    boxes.swap(merged);
  }
  // -------------------------------------------------------------

  // Image Callback ----------------------------------------------
  void imageCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv::Mat bgr;
    try {
      bgr = cv_bridge::toCvShare(msg, "bgr8")->image; // copy the image streamed in for processing
    } catch (const cv_bridge::Exception& e) { // send message if this fails (cv not working nicely with ros)
      RCLCPP_WARN(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    if (bgr.empty()) return; // get out if the image is empty

    cv::Mat gray; cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY); // grayscale the image to make finding the dark regions easier
    cv::Mat th; // binary mask for the image

    // Thresholding -------------------------------
    // Simple mode tailored to black marker on a whiteboard for practical testing uses Otsu ===================================
    if (mode == "simple") {
      cv::Mat blur; 
      cv::GaussianBlur(gray, blur, cv::Size(5,5), 0);

      // Run Otsu to find best threshold, then subtract bias
      cv::Mat tmp;
      double otsu = cv::threshold(blur, tmp, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
      double thr = std::clamp(otsu - static_cast<double>(simple_bias), 0.0, 255.0);
      cv::threshold(blur, th, thr, 255, cv::THRESH_BINARY_INV);

      // Strong morphology for solid blob
      int ck = std::max(9, close_k | 1);
      cv::morphologyEx(th, th, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {ck, ck})); // Close up boxes to prevent a bunch of little "speckles" of potholes
      int mpx = std::max(1, simple_merge_px);
      cv::dilate(th, th, cv::getStructuringElement(cv::MORPH_ELLIPSE, {mpx, mpx}));
      cv::erode (th, th, cv::getStructuringElement(cv::MORPH_ELLIPSE, {mpx, mpx}));
    }
    // ==========================================
    
    // Blackhat mode for asphalt ===================
    else {
      cv::Mat blur; cv::bilateralFilter(gray, blur, 5, 50, 50); // makes the image less noisy (good for uneven texture/coloring of asphalt) and keep the edges
      int k = std::max(3, bh_kernel | 1);
      cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, {k, k}); // hilights the dark depressions of the pothole
      cv::Mat blackhat; cv::morphologyEx(blur, blackhat, cv::MORPH_BLACKHAT, se);
      int block = std::max(3, bh_block | 1);
      int C = bh_c; // constant
      cv::adaptiveThreshold(blackhat, th, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block, C); // Local threshold (biased with the constant for dark/black regions) more eager to choose dark with Gaussian
      int ck = std::max(3, close_k | 1);
      cv::morphologyEx(th, th, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {ck, ck}));
      // cleaning up by lightly dilating and eroding (fine tuning) 
      cv::dilate(th, th, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3,3}));
      cv::erode (th, th, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3,3}));
    }
    // ============================================
    // ----------------------------------------------------------------------

    // Extract boxes ---------------------------------------------------------
    std::vector<cv::Rect> boxes;
    if (use_cc) { // if we see boxes that have the conditions of being able to be connected
      cv::Mat labels, stats, cents;
      int n = cv::connectedComponentsWithStats(th, labels, stats, cents, 8, CV_32S); // guarantees one detection per connected blob
      for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_area || area > max_area) continue;
        cv::Rect box(stats.at<int>(i, cv::CC_STAT_LEFT),
                     stats.at<int>(i, cv::CC_STAT_TOP),
                     stats.at<int>(i, cv::CC_STAT_WIDTH),
                     stats.at<int>(i, cv::CC_STAT_HEIGHT));
        boxes.push_back(box);
      }
    } 
    else { // in blackhat mode we use findContours() instead of connectedCompenentsWithStats()
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(th, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_area || area > max_area) continue;
        double rnd = roundnessFromContour(c);
        if (rnd < round_min) continue;
        boxes.push_back(cv::boundingRect(c));
      }
    }

    if (merge_boxes) mergeNearbyBoxes(boxes, bgr.size()); // merging boxes more for extra guard against the speckles/fragmenting
    // ----------------------------------------------------------------------

    // Publishing ------------------------------------------------------------
    vision_msgs::msg::Detection2DArray dets;
    dets.header = msg->header;
    cv::Mat overlay = bgr.clone();

    for (const auto& box : boxes) {
      vision_msgs::msg::Detection2D d;
      d.header = dets.header;
      // dimension of the bounding box
      d.bbox.center.position.x = box.x + box.width  * 0.5;
      d.bbox.center.position.y = box.y + box.height * 0.5;
      d.bbox.size_x = box.width;
      d.bbox.size_y = box.height;

      vision_msgs::msg::ObjectHypothesisWithPose hyp; // publish a vision message
      hyp.hypothesis.class_id = "pothole"; // id of the message
      hyp.hypothesis.score = 0.5; // relative confidence (just a stub for now could make it a function of blob roundness/area/darkness)
      d.results.push_back(hyp);

      dets.detections.push_back(d);
      cv::rectangle(overlay, box, {0,255,0}, 2);
    }

    detections_pub->publish(dets); // publish out 

    std_msgs::msg::Float32 conf; // confidence of detected pothole
    conf.data = std::min(1.0f, static_cast<float>(boxes.size()) / 10.0f);
    conf_pub->publish(conf);

    if (dbg) { // debug image for testing that shows the masked image and bounding boxes
      cv::Mat out_img;
      if (debug_view == "overlay") {
        out_img = overlay;
      } 
      else if (debug_view == "mask") {
        cv::cvtColor(th, out_img, cv::COLOR_GRAY2BGR);
      } 
      else {
        cv::Mat mask_bgr; cv::cvtColor(th, mask_bgr, cv::COLOR_GRAY2BGR);
        cv::hconcat(overlay, mask_bgr, out_img);
      }
      auto out = cv_bridge::CvImage(msg->header, "bgr8", out_img).toImageMsg();
      dbg_pub.publish(out);
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 1000,
      "mode=%s boxes=%zu bias=%d close=%d", mode.c_str(), boxes.size(), simple_bias, close_k);
  }

  // Members --------------------
  std::string cam_topic;
  int min_area, max_area;
  double round_min;
  bool dbg;

  // Processing ----------------------------
  std::string mode;
  int bh_kernel, bh_block, bh_c;
  int simple_bias, simple_merge_px;
  int close_k;
  bool use_cc, merge_boxes;
  int pad_px;
  double iou_merge;
  std::string debug_view;

  // IO ------------------------------------
  image_transport::Subscriber sub;
  image_transport::Publisher dbg_pub;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr conf_pub;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PotholeDetector>()); // spin up the node
  rclcpp::shutdown();
  return 0;
}
