# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

# This should be used only for rosbags taken with R1 when extra_camera_tf is not present

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=[
                '0.0', '0.0', '0.0',        #x, y, z
                '0.0', '0.0', '0.0', '1.0',  #qx qy qz qw
                'depth_center',         
                'extra_camera_tf'        
            ],
            output='screen'
            )
    ])
