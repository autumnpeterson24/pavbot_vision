cmake_minimum_required(VERSION 3.10)
project(lane_lab)

# Find required packages 
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Mock camera node
add_executable(mock_camera src/mock_camera.cpp)
ament_target_dependencies(mock_camera
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  OpenCV
)

# Lane detector node
add_executable(lane_detector src/lane_detector.cpp)
ament_target_dependencies(lane_detector
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  image_transport
  cv_bridge
  OpenCV
)

# Install targets (These are the nodes for the mock cam and lane detection)
install(TARGETS
  mock_camera
  lane_detector
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config files
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
