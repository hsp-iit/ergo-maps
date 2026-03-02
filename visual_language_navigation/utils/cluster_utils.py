# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

from scipy.spatial import ConvexHull
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import numpy as np
from visual_language_navigation.utils.conversion_utils import convert_from_costmap_cell

def extract_cluster_boundaries(cluster_points):
    try:
        chull = ConvexHull(points=cluster_points)
        return cluster_points[chull.vertices]
    except Exception as ex:
        print(f"{ex=}")
        return None

def compute_cluster_centers(points, aligned_labels, cell_size, grid_size):
        # Rotation matrix for aligning the semantic map with the nav2 map
        rot_mat_z = np.array([[-1, 0, 0],
                              [0, -1, 0],
                              [0, 0, 1]])
        unique_labels, counts = np.unique(aligned_labels, return_counts=True)    # -1 label means no cluster
        # Compute clusters center and hulls
        centers = []
        hulls_verticies = []
        for label in unique_labels:
            if label >= 0:
                cluster_points = np.array(points[aligned_labels == label])
                points_num = len(cluster_points)
                if points_num == 0:
                    continue
                # Hulls verticies computation
                verticies = extract_cluster_boundaries(cluster_points)
                if verticies is not None:
                    hulls_verticies.append(verticies * cell_size  - (grid_size * cell_size / 2))
                    for i in range(len(hulls_verticies[-1])):
                        hulls_verticies[-1][i] = rot_mat_z[:2, :2] @ (hulls_verticies[-1][i]).T
                
                # Compute center average
                x = np.sum(cluster_points[:, 0]) / points_num
                y = np.sum(cluster_points[:, 1]) / points_num
                centers.append([x,y])
        return np.array(centers), counts[1:], hulls_verticies
                
    
def pub_clusters_verticies(hulls_list, map_frame, stamp, bbox_marker_pub):
    hull_marker = Marker()
    hull_marker.header.frame_id = map_frame
    hull_marker. header.stamp = stamp
    hull_marker.id = 11
    hull_marker.type = Marker.LINE_LIST
    hull_marker.action = Marker.ADD
    hull_marker.scale.x = hull_marker.scale.y = hull_marker.scale.z = 0.05
    c = ColorRGBA()
    c.a = 1.0
    c.r = 0.0
    c.g = 0.0
    c.b = 1.0
    hull_marker.color = c
    hull_marker.pose.orientation.x = 0.0
    hull_marker.pose.orientation.y = 0.0
    hull_marker.pose.orientation.z = 0.0
    hull_marker.pose.orientation.w = 1.0
    hull_marker.pose.position.x = 0.0
    hull_marker.pose.position.y = 0.0
    hull_marker.pose.position.z = 0.2

    for polygon_verticies in hulls_list:
        # Vertexes of the rectangle
        for i in range(len(polygon_verticies)):
            p1 = Point(x=polygon_verticies[i][0],
                       y=polygon_verticies[i][1])
            p2 = Point(x=polygon_verticies[(i+1) % len(polygon_verticies)][0],
                       y=polygon_verticies[(i+1) % len(polygon_verticies)][1])
            hull_marker.points.append(p1)
            hull_marker.points.append(p2)

    bbox_marker_pub.publish(hull_marker)

# Publishing a red marker where the center of the closest cluster is located, and a green marker where the closest free spot has been found
def publish_goal_markers(closest_center, map_frame, stamp, global_costmap_msg, goal_marker_pub, free_cell = None):
    marker = Marker()
    marker.id = 0
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.type = marker.POINTS
    marker.action = marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.1
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.2
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    p = Point()
    p.x = closest_center[0]
    p.y = closest_center[1]
    p.z = 0.20  # height
    marker.points.append(p)
    c = ColorRGBA()
    c.r, c.b, c.g  = np.array([1.0, 0.0, 0.0])  # In red we publish the marker in the center of the cluster
    c.a = 0.9
    marker.colors.append(c)
    if free_cell is not None:
        p = Point()
        p.x, p.y = convert_from_costmap_cell(free_cell[0], free_cell[1], global_costmap_msg)
        p.z = 0.20  # height
        marker.points.append(p)
        col = ColorRGBA()
        col.r, col.b, col.g  = np.array([0.0, 0.0, 1.0])    # In green we publish the marker in the closest free cell to the marker
        col.a = 0.9
        marker.colors.append(col)
    goal_marker_pub.publish(marker)