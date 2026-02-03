cmake_minimum_required(VERSION 3.10)
project(pavbot_vision)

# Find required packages 
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)




# Actual camera publisher ---------------------------
add_executable(cam_pub src/cam_pub.cpp)
ament_target_dependencies(cam_pub 
  rclcpp image_transport 
  cv_bridge 
  sensor_msgs
  )
target_link_libraries(cam_pub ${OpenCV_LIBS})


# Pothole detector dual node ------------------------------
add_executable(pothole_detector_dual src/pothole_detector_dual.cpp)
ament_target_dependencies(pothole_detector_dual
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  image_transport
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Pothole detector node ------------------------------
add_executable(pothole_detector src/pothole_detector.cpp)
ament_target_dependencies(pothole_detector
  rclcpp 
  image_transport 
  cv_bridge 
  OpenCV 
  sensor_msgs 
  std_msgs 
  vision_msgs
  )

target_compile_features(pothole_detector PUBLIC cxx_std_17)


# Lane detector node -------------------------------
add_executable(lane_detector_dual src/lane_detector_dual.cpp)
ament_target_dependencies(lane_detector_dual
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  image_transport
  cv_bridge
  OpenCV
  visualization_msgs
)

# Mock camera node (Not really used anymore)
# add_executable(mock_camera src/mock_camera.cpp)
# ament_target_dependencies(mock_camera
#   rclcpp
#   sensor_msgs
#   image_transport
#   cv_bridge
#   OpenCV
# )

# Install targets

install(TARGETS
  #mock_camera

  cam_pub
  lane_detector_dual
  pothole_detector
  pothole_detector_dual

  DESTINATION lib/${PROJECT_NAME}
)


# Install launch, config, and media (don't need media for debugging anymore now with real cams)
install(DIRECTORY 
  launch 
  config 
  #media
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
