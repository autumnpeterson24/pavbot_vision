"""
REAL PHYSICAL VISION WITH DUAL CAMERAS LAUNCH (Launches dual lane detection and the camera publishers)

Launch command:
ros2 launch physical_vision.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_lane_detector = LaunchConfiguration("use_lane_detector")
    use_pothole_detector = LaunchConfiguration("use_pothole_detector")

    cams_yaml = PathJoinSubstitution([
        FindPackageShare("pavbot_vision"),
        "config",
        "dual_cam_pub.yaml",
    ])

    lane_yaml = PathJoinSubstitution([
        FindPackageShare("pavbot_vision"),
        "config",
        "lane_detector_dual_real.yaml",
    ])

    pothole_yaml = PathJoinSubstitution([
        FindPackageShare("pavbot_vision"),
        "config",
        "pothole_detector_real.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("use_lane_detector", default_value="true"),
        DeclareLaunchArgument("use_pothole_detector", default_value="true"),

        # Static TFs for physical camera mounts (Euler = Yaw, Pitch, Roll!!! ) ===============
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0.55","0.3","1.0", # X (forward, back), Y (left, right), Z (Up, down)
                       "0.31","0.52","0.0", # Yaw, Pitch, Roll | -0.1, 0.5, 0.0
                       "base_link","left_camera_link"],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0.55","-0.3","1.0", # X (forward, back), Y (left, right), Z (Up, down)
                       "-0.31","0.52","0.0", # Yaw, Pitch, Roll | 0.1 0.5 0.0
                       "base_link","right_camera_link"],
            output="screen",
        ),


        # Dual camera publisher ==========
        Node(
            package="pavbot_vision",
            executable="dual_cam_pub",
            name="dual_cam_pub",
            output="screen",
            parameters=[cams_yaml],
        ),

        # Dual lane detector ============
        Node(
            package="pavbot_vision",
            executable="lane_detector_dual",
            name="lane_detector_dual",
            condition=IfCondition(use_lane_detector),
            output="screen",
            parameters=[lane_yaml],
        ),

        # Pothole detector (physical) =========
        Node(
            package="pavbot_vision",
            executable="pothole_detector_single",
            name="pothole_detector_single",
            condition=IfCondition(use_pothole_detector),
            output="screen",
            parameters=[pothole_yaml],
        ),


        # Pothole points bridge (publish /potholes/points for costmap) =====
        Node(
            package="pavbot_vision",
            executable="pothole_points_bridge",
            name="pothole_points_bridge",
            output="screen",
            parameters=[{
                "use_sim_time": False,

            }],
        ),

        # static transform for LiDAR =================
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="lidar_frame_fix",
            output="screen",
            arguments=[
                "0.41", "0.000000", "0.70",
                "3.14159", "0.0", "0.0",
                "base_link", "lidar_link/lidar"
            ],
            parameters=[{"use_sim_time": False}],
            )
    ])