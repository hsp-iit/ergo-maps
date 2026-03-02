# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import numpy as np

def transform_in_robot_frame(robot_pose, object_string : str, object_list):
    """
    Converts the poses contained in object_list in the robot perspective.

    :param robot_pose: a vector (x_robot, y_robot, theta) containing the pose of the robot on the plane. x_robot and y_robot are the traslation components, while theta is the orientation of the robot on the plane.
    :param object_string: the string of the object class.
    :param object_list: a vector of 2D poses (x_object, y_object), of shape (N, 2), to be converted in the robot perspective.
    :return: 1) a Tuple containing the object_string as first object, representing the name of the objects that are being converted, and 2) a list of 2D poses (x_object, y_object) of the object class expressed in the robot perspective.
    """
    #TODO sanity checks for robot pose and poses
    r_pose = np.array(robot_pose)
    rot_matrix = np.array([[np.cos(-r_pose[2]), - np.sin(-r_pose[2])],
                          [np.sin(-r_pose[2]), np.cos(-r_pose[2])]])
    poses = np.array(object_list)
    transformed_poses = []
    #if poses.shape[0] > 1:
    for pose in poses:
        #Translate first
        pose[0] = pose[0] - r_pose[0]
        pose[1] = pose[1] - r_pose[1]
        # Rotation
        final_pose = rot_matrix @ pose[0:2]
        transformed_poses.append(list(final_pose))

    return object_string, transformed_poses
    
def transform_from_robot_frame(robot_pose, object_string : str, object_list):
    """
    The opposite of transform_in_robot_frame.
    Converts back the poses contained in object_list from the robot perspective to the global reference.

    :param robot_pose: a vector (x_robot, y_robot, theta) containing the pose of the robot on the plane. x_robot and y_robot are the traslation components, while theta is the orientation of the robot on the plane.
    :param object_string: the string of the object class.
    :param object_list: a vector of N 2D poses (x_object, y_object), of shape (N, 2), expressed in the robot perspective.
    :return: 1) a Tuple containing the object_string as first object, representing the name of the objects that are being converted, and 2) a list of 2D poses (x_object, y_object) of the object class expressed in the global reference.
    """
    #TODO sanity checks for robot pose and poses
    r_pose = np.array(robot_pose)
    rot_matrix = np.array([[np.cos(r_pose[2]), - np.sin(r_pose[2])],
                          [np.sin(r_pose[2]), np.cos(r_pose[2])]])
    poses = np.array(object_list)
    transformed_poses = []
    if poses.ndim > 1:
        for pose in poses:
            # Rotation first
            rotated_pose = rot_matrix @ pose[0:2]
            if len(pose) == 3:
                pose[2] = pose[2] + r_pose[2]   
            #Translate
            pose[0] = rotated_pose[0] + r_pose[0]
            pose[1] = rotated_pose[1] + r_pose[1]
            
            transformed_poses.append(pose)
    else:
        # Rotation
        rotated_pose = rot_matrix @ poses[0:2]
        #Translate
        poses[0] = rotated_pose[0] + r_pose[0]
        poses[1] = rotated_pose[1] + r_pose[1]
        if len(poses) == 3:
                poses[2] = poses[2] + r_pose[2]
        
        transformed_poses.append(poses)
    return object_string, transformed_poses

def get_distance(pos1, pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    return str(np.linalg.norm(pos1 - pos2))