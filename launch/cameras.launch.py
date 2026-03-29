from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare("pavbot_vision"),
        "config",
        "dual_cam_pub.yaml"
    ])

    return LaunchDescription([
        Node(
            package="pavbot_vision",
            executable="dual_cam_pub",
            name="dual_cam_pub",
            output="screen",
            parameters=[params]
        )
    ])