# PAVbot Perception System (ROS 2 Humble)

[![ROS 2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Build](https://img.shields.io/badge/Build-Colcon-success.svg)](https://colcon.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Overview

This package provides **lane and pothole detection** for **PAVbot** (Pathfinding Autonomous Vehicle) competing in the **2026 IGVC AutoNav Challenge**.  
Both detectors run in real time on embedded Jetson hardware using **pure OpenCV pipelines**.
---

### ROS2 Nodes
## Publish (->), Subscribe (<-)

| **`lane_detector`** Detects lane markings and publishes a smoothed centerline path.| ->`/lanes/centerline` (`nav_msgs/Path`)-> `/lanes/confidence` (`std_msgs/Float32`)-> `/lanes/debug_image` (`sensor_msgs/Image`)<- `/lane_cam/image_raw`
| **`pothole_detector`** Detects dark concave regions resembling potholes.| -> `/potholes/detections` (`vision_msgs/Detection2DArray`) -> `/potholes/confidence` (`std_msgs/Float32`)->v`/potholes/debug_image` (`sensor_msgs/Image`) <- `/pothole_cam/image_raw`
| **`cam_pub`** Publishes live USB camera feeds for each perception module.| ->`/lane_cam/image_raw`, `/pothole_cam/image_raw`

---

## Algorithms

### **Lane Detection**
HSV threshold -> Sobel edges -> sliding-window quadratic fit -> centerline generation.  
Publishes a `nav_msgs/Path` for the navigation stack and a confidence metric for lane visibility.

### **Pothole Detection**
Two modes selected via `pothole_detector.mode` parameter:

| `blackhat` | Morphological black-hat filter + adaptive Gaussian threshold to isolate dark depressions on asphalt. **Outdoor IGVC testing**
| `simple` | Otsu threshold with bias + morphological closing for black circles on white backgrounds. **For testing**

Each blob is filtered by area and roundness, optionally merged, and published as `vision_msgs/Detection2D` boxes.

---

## Dependencies

```bash
sudo apt update
sudo apt install -y ros-humble-desktop \
  ros-humble-image-transport ros-humble-cv-bridge \
  ros-humble-vision-msgs ros-humble-camera-info-manager \
  ros-humble-rviz2 libopencv-dev v4l-utils \
  python3-rosdep
sudo rosdep init 2>/dev/null || true
rosdep update
