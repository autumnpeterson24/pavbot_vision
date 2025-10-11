from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare('lane_lab'), 'config', 'lane.yaml'])

    return LaunchDescription([
        Node(
            package='lane_lab',
            executable='mock_camera',
            name='mock_camera',
            parameters=[cfg],
            output='screen'
        ),
        Node(
            package='lane_lab',
            executable='lane_detector',
            name='lane_detector',
            parameters=[cfg],
            output='screen'
        )
    ])
