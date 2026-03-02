# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the directory where the package is installed
    package_dir = get_package_share_directory('visual_language_navigation')

    # Build the full path to your parameter file
    rviz_config = os.path.join(package_dir, 'params/rviz', 'vlnav.rviz')

    return LaunchDescription([
        Node(package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output={'both': 'log'}
            )
    ])