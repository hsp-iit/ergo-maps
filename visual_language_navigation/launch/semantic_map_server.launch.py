# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import os
from launch import LaunchDescription

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the directory where the package is installed
    package_dir = get_package_share_directory('visual_language_navigation')

    # Build the full path to your parameter file
    param_file = os.path.join(package_dir, 'params', 'semantic_map_server_params.yaml')

    # Define the node to launch
    semantic_map_server_node = Node(
        package='visual_language_navigation',
        executable='semantic_map_server',   # defined in setup.py entry_points
        name='semantic_map_server',
        output='screen',
        parameters=[param_file],
        emulate_tty=True,  # allows colored logs and line-by-line printing
    )

    return LaunchDescription([
        semantic_map_server_node
    ])