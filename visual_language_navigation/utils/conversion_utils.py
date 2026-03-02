# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import numpy as np
import math
from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2, Image

def quaternion_matrix(quaternion):  #Copied from https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py#L1515
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def quaternion_from_euler(ai, aj, ak):  # From https://docs.ros.org/en/iron/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss    #w

    return q

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into roll, pitch, yaw
    """
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def xyzrgb_array_to_pointcloud2(points, colors, stamp, frame_id, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points and a synched array of color values.
        '''

        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzrgb')]
        nbytes = 6
        xyzrgb = np.array(np.hstack([points, colors/255]), dtype=np.float32)
        #xyzrgb = np.array(points_ren, dtype=np.float32)
        msg = PointCloud2(header=header, 
                          height = 1, 
                          width= points.shape[0], 
                          fields=fields, 
                          is_dense= False, 
                          is_bigedian=False, 
                          point_step=(itemsize * nbytes), 
                          row_step = (itemsize * nbytes * points.shape[0]), 
                          data=xyzrgb.tobytes())

        return msg

def numpy_to_ros2_image(rgb: np.ndarray, stamp, frame_id="camera_rgb_frame") -> Image:
    msg = Image()
    msg.height = rgb.shape[0]
    msg.width = rgb.shape[1]
    msg.encoding = 'rgba8'
    msg.is_bigendian = False
    msg.step = rgb.shape[1] * 3
    msg.data = rgb.tobytes()

    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    return msg

def convert_in_costmap_cell(pose_x, pose_y, global_costmap_msg):
    cell_x = int((pose_x - global_costmap_msg.info.origin.position.x) / global_costmap_msg.info.resolution)
    cell_y = int((pose_y - global_costmap_msg.info.origin.position.y) / global_costmap_msg.info.resolution)
    return cell_x, cell_y
    
def convert_from_costmap_cell(cell_x, cell_y, global_costmap_msg):
    pose_x = cell_x * global_costmap_msg.info.resolution + global_costmap_msg.info.origin.position.x
    pose_y = cell_y * global_costmap_msg.info.resolution + global_costmap_msg.info.origin.position.y
    return pose_x, pose_y