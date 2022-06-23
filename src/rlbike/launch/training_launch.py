from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rlbike',
            node_executable='run',
            node_name='node_rl',
            output='screen',
            emulate_tty=True,
            parameters=[
                {"frames": 50000}
            ]
        ),
        Node(  
            package='rlbike',
            node_executable='motor_bridge',
            node_name='motor_bridge',
            output='screen',
            emulate_tty=True
        ),
        Node(
            package='rlbike',
            node_executable='imu_bridge',
            node_name='imu_bridge',
            output='screen',
            emulate_tty=True
        )
    ])

