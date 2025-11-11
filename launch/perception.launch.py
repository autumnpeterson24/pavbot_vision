"""
Launch file for lane_detector and pothole_detector nodes and cam_pub to link together
everything including camera hardware. Makes it easier to test.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    lane_device_arg    = DeclareLaunchArgument('lane_device', default_value='/dev/video3')
    pothole_device_arg = DeclareLaunchArgument('pothole_device', default_value='/dev/video7')


    # Lane camera node for lane detection
    lane_cam = Node( 
        package='lane_lab', executable='cam_pub',
        namespace='lane_cam', name='lane_cam_pub', output='screen',
        parameters=[
            {'device': LaunchConfiguration('lane_device')},
            {'width': 1280}, {'height': 720}, {'fps': 30.0}, {'fourcc': 'MJPG'},
            {'frame_id': 'lane_cam_optical_frame'}  # optional
        ],
        # REMAP the absolute topics to namespaced topics
        remappings=[
            ('/camera/image_raw',   '/lane_cam/image_raw'),
            ('/camera/camera_info', '/lane_cam/camera_info')
        ],
        # respawn=True,
    )

# pothole camera node for pothole detection
    pothole_cam = Node(
        package='lane_lab', executable='cam_pub',
        namespace='pothole_cam', name='pothole_cam_pub', output='screen',
        parameters=[
            {'device': LaunchConfiguration('pothole_device')},
            {'width': 1280}, {'height': 720}, {'fps': 30.0}, {'fourcc': 'MJPG'},
            {'frame_id': 'pothole_cam_optical_frame'}  # optional
        ],
        remappings=[
            ('/camera/image_raw',   '/pothole_cam/image_raw'),
            ('/camera/camera_info', '/pothole_cam/camera_info')
        ],
        # respawn=True,
    )

    lane_node = Node(
        package='lane_lab', executable='lane_detector', name='lane_detector', output='screen',
        parameters=[{'camera_topic': '/lane_cam/image_raw'}], # Use correct param name: declare_parameter<std::string>("camera_topic", "/camera/image_raw")
        remappings=[('/camera/image_raw', '/lane_cam/image_raw')]
    )

    pothole_node = Node(
        package='lane_lab', executable='pothole_detector', name='pothole_detector', output='screen',
        parameters=[
            {'image_topic': '/pothole_cam/image_raw'},   # pothole node expects this param
            {'min_area': 50}, {'max_area': 150000},
            {'roundness_min': 0.2}, {'debug': True}
        ]
    )

    return LaunchDescription([
        lane_device_arg, pothole_device_arg,
        lane_cam, pothole_cam, lane_node, pothole_node
    ])
