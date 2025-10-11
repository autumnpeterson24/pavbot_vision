# PAVbot Lane Detection (ROS 2 Humble)

This package implements a **prototype lane detection system** for the PAVbot platform competing in the **2026 IGVC AutoNav Challenge**.

It processes video or live camera data using OpenCV and publishes lane centerlines as ROS 2 topics that can be used for autonomous navigation.

*Still need to refine the poses and coordinate publishing to make sure the coordinates are actually correct within the world space and 
I want to add Gazebo implementation as well.

---

## Overview

**Nodes:**
- `lane_detector` – Subscribes to `/camera/image_raw`, processes frames, and publishes:
  - `/lanes/centerline` (`nav_msgs/Path`) — estimated lane center path
  - `/lanes/confidence` (`std_msgs/Float32`) — lane detection confidence
  - `/lanes/debug_image` (`sensor_msgs/Image`) — visual overlay for RViz

- `mock_camera` – Streams a video as fake camera input (useful for testing without hardware)

---

## Dependencies

- ROS 2 Humble
- `image_transport`
- `cv_bridge`
- `OpenCV`
- `nav_msgs`, `geometry_msgs`, `std_msgs`

Install (if not already):
```bash
sudo apt install ros-humble-image-transport ros-humble-cv-bridge
```
---
## Build Instructions
```bash
cd ~/ros2_ws/src
git clone https://github.com/autumnpeterson24/PAVbot_lane_detection.git
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Run the offline Demo
ros2 launch lane_lab offline_lane_demo.launch.py mock_camera.video_path:=/home/jetson/Videos/lanes_demo.mp4

---

## Visualize in RViz
rviz2 -d ~/ros2_ws/src/lane_lab/config/lane_view.rviz
