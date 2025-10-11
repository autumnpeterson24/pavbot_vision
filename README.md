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

Install ROS 2 Humble + build tools (only have to do this once):
```bash
# Basic dev tools
sudo apt update
sudo apt install -y curl gnupg lsb-release build-essential

# ROS 2 Humble apt repo + key
sudo mkdir -p /usr/share/keyrings
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
 | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
https://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update

# Full desktop (rviz2, tools, colcon, etc.)
sudo apt install -y ros-humble-desktop

# Extra packages used by the repo
sudo apt install -y ros-humble-image-transport ros-humble-cv-bridge \
                    ros-humble-camera-info-manager ros-humble-rviz2 \
                    libopencv-dev v4l-utils
# ! If this is the first ROS 2 machine you’ve ever used, set up rosdep: !
sudo apt install -y python3-rosdep
sudo rosdep init 2>/dev/null || true
rosdep update
```
---
## Set up ros2_ws: 
**Create workspace and clone in the pkg from the repo**
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone repo here
git clone https://github.com/autumnpeterson24/PAVbot_lane_detection.git lane_lab

```
**Install Dependencies**
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```
**Build and Source**
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Run the offline Demo
```bash
ros2 launch lane_lab offline_lane_demo.launch.py
```

---

## Visualize in RViz (in another terminal)
```bash
rviz2 -d ~/ros2_ws/src/lane_lab/config/lane_view.rviz
```
