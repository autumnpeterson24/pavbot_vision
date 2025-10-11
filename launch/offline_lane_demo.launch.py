import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('lane_lab')

    # Resolve paths inside the installed package
    cfg_path   = os.path.join(pkg_share, 'config', 'lane.yaml')
    video_path = os.path.join(pkg_share, 'media',  'RightWhiteLane.mp4')

    mock_camera = Node(
        package='lane_lab',
        executable='mock_camera',
        name='mock_camera',
        parameters=[{
            'video_path': video_path,        # uses package media file
            'fps': 30.0,
            'loop': True,
            'frame_id': 'camera_optical_frame'
        }],
        output='screen'
    )

    lane_detector = Node(
        package='lane_lab',
        executable='lane_detector',
        name='lane_detector',
        parameters=[cfg_path],              # uses package config file
        output='screen'
    )

    return LaunchDescription([mock_camera, lane_detector])
