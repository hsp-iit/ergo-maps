# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from multiprocessing import Process
import matplotlib.patches as mpatches
from PIL import Image
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Publisher
from builtin_interfaces.msg import Time

class Visualizer(Process):

    def __init__(self, pcd: o3d.geometry.PointCloud, name: str):
        super().__init__() 
        self.pcd = pcd
        self.name = name

    def run(self):
        o3d.visualization.draw_geometries([self.pcd], window_name=self.name)

def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray, window_name: str, voxel_size=1.0):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=1.0)
    o3d.visualization.draw_geometries_with_vertex_selection([pcd])
    # visualize the point cloud
    #Visualizer(voxel_grid, window_name).start()

def compute_point_cloud_difference(pcd_1, pcd_2, distance_threshold = 0.01):
    """
    Computes the difference between two point clouds and colors the difference points in red.
    
    Args:
    pcd_1 (np.ndarray): The first point cloud.
    pcd_2 (np.ndarray): The second point cloud.
    distance_threshold (float): The distance threshold for determining point differences.
    
    Returns:
    o3d.geometry.PointCloud: A point cloud containing points in pcd_1 that are not in pcd_2, colored in red.
    """
    # Convert point clouds to numpy arrays
    points_1 = pcd_1
    points_2 = pcd_2
    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd_2)
    # Create a KDTree for the second point cloud
    pcd_2_tree = o3d.geometry.KDTreeFlann(pcd2_o3d)
    
    # List to hold the points that are different
    diff_points = []
    
    # Check each point in pcd_1
    for point in points_1:
        [_, idx, _] = pcd_2_tree.search_knn_vector_3d(point, 1)
        if np.linalg.norm(points_2[idx[0]] - point) < distance_threshold:
            diff_points.append(point)
    
    # Create a new point cloud with the different points
    diff_pcd = o3d.geometry.PointCloud()
    diff_pcd.points = o3d.utility.Vector3dVector(np.array(diff_points))
    
    # Color the different points in red
    colors = np.array([[1, 0, 0] for _ in diff_points])  # RGB color for red
    diff_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return diff_pcd

def get_heatmap_from_mask_3d(
    pc: np.ndarray, mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01
) -> np.ndarray:
    target_pc = pc[mask, :]
    other_ids = np.where(mask == 0)[0]
    other_pc = pc[other_ids, :]

    target_sim = np.ones((target_pc.shape[0], 1))
    other_sim = np.zeros((other_pc.shape[0], 1))
    pbar = tqdm(other_pc, desc="Computing heat", total=other_pc.shape[0])
    for other_p_i, p in enumerate(pbar):
        dist = np.linalg.norm(target_pc - p, axis=1) / cell_size
        min_dist_i = np.argmin(dist)
        min_dist = dist[min_dist_i]
        other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

    new_pc = pc.copy()
    heatmap = np.ones((new_pc.shape[0], 1), dtype=np.float32)
    for s_i, s in enumerate(other_sim):
        heatmap[other_ids[s_i]] = s
    return heatmap.flatten()


def visualize_masked_map_3d(pc: np.ndarray, mask: np.ndarray, rgb: np.ndarray, transparency: float = 0.5):
    heatmap = mask.astype(np.float16)
    visualize_heatmap_3d(pc, heatmap, rgb, transparency)


def visualize_heatmap_3d(pc: np.ndarray, heatmap: np.ndarray, rgb: np.ndarray, window_name: str = "heatmap 3d", transparency: float = 0.5):
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat.reshape(-1, 3)[:, ::-1].astype(np.float32)
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_3d(pc, heat_rgb, window_name)


def pool_3d_label_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d


def pool_3d_rgb_to_2d(rgb: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    rgb_2d = np.zeros((gs, gs, 3), dtype=np.uint8)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            rgb_2d[row, col] = rgb[i]

    return rgb_2d


def get_heatmap_from_mask_2d(mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01) -> np.ndarray:
    dists = distance_transform_edt(mask == 0) / cell_size
    tmp = np.ones_like(dists) - (dists * decay_rate)
    heatmap = np.where(tmp < 0, np.zeros_like(tmp), tmp)

    return heatmap


def visualize_rgb_map_2d(rgb: np.ndarray):
    """visualize rgb image

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
    """
    rgb = rgb.astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    cv2.imshow("rgb map", bgr)
    cv2.waitKey(0)


def visualize_heatmap_2d(rgb: np.ndarray, heatmap: np.ndarray, transparency: float = 0.5):
    """visualize heatmap

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        heatmap (np.ndarray): (gs, gs) element range [0, 1] np.float32
    """
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32)  # convert to RGB
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_2d(heat_rgb)


def visualize_masked_map_2d(rgb: np.ndarray, mask: np.ndarray):
    """visualize masked map

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        mask (np.ndarray): (gs, gs) element range [0, 1] np.uint8
    """
    visualize_heatmap_2d(rgb, mask.astype(np.float32))


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None, ignore_ids_list=[]):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[int(index)]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)

    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete

# Made for debugging costmap
def pub_costmap_markers(costmap_grid_bool, map_frame : str, stamp : Time, global_costmap_msg : OccupancyGrid, goal_marker_pub : Publisher):
    """
    Publish free cells from a costmap as green points in RViz.

    Each free cell in the boolean grid is visualized as a small green point marker.
    Used only for debug.

    Args:
        costmap_grid_bool (np.ndarray):
            2D boolean numpy array representing the costmap grid.
            True = occupied/blocked, False = free.
        map_frame (str):
            Frame ID in which the costmap is expressed (e.g., "map").
        stamp (builtin_interfaces.msg.Time):
            Timestamp for the marker header.
        global_costmap_msg (nav_msgs.msg.OccupancyGrid):
            Original occupancy grid message, used to retrieve resolution and origin.
        goal_marker_pub (rclpy.publisher.Publisher):
            ROS2 publisher for `visualization_msgs/Marker`.
    """
    marker = Marker()
    marker.id = 0
    marker.header.frame_id = map_frame
    marker.header.stamp = stamp
    marker.type = marker.POINTS
    marker.action = marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 0.05
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    c = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)
    for i in range(costmap_grid_bool.shape[0]):
        for j in range(costmap_grid_bool.shape[1]):
            if not costmap_grid_bool[i][j]:    # Print only the free cells
                p = Point()
                # Proper way to interpret costmap from message and stacking it self.costmap_grid = np.array(self.global_costmap_msg.data).reshape((self.global_costmap_msg.info.height, self.global_costmap_msg.info.width)).T
                p.x = i * global_costmap_msg.info.resolution + global_costmap_msg.info.origin.position.x
                p.y = j * global_costmap_msg.info.resolution + global_costmap_msg.info.origin.position.y
                p.z = 0.20  # height of the markers
                marker.points.append(p)
                marker.colors.append(c)
    goal_marker_pub.publish(marker)

def pub_bounding_boxes(b_boxes, map_frame : str, stamp : Time, bbox_marker_pub : Publisher, id = 11):
    """
    Publish 2D bounding boxes as a LINE_LIST marker in RViz.

    Each bounding box is expected to be defined by 4 corner points in 2D (x, y).
    The function draws the box as connected line segments and publishes them as
    a `visualization_msgs/Marker`.

    Args:
        b_boxes (list of list of tuple):
            List of bounding boxes, where each bounding box is a list of 4 corner
            points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        map_frame (str):
            Frame ID in which the bounding boxes are expressed ( "map").
        stamp (builtin_interfaces.msg.Time):
            Timestamp for the marker header.
        bbox_marker_pub (rclpy.publisher.Publisher):
            ROS2 publisher for `visualization_msgs/Marker` messages.
        id (int, optional):
            Marker ID, used for uniquely identifying the marker in RViz.
            Defaults to 11.

    Notes:
        - Bounding boxes are drawn at a fixed z-height (0.1 m).
        - Marker color is fixed to solid red (RGBA = [1, 0, 0, 1]).
        - Each box is rendered as four connected line segments.
    """
    bb_marker = Marker()
    bb_marker.header.frame_id = map_frame
    bb_marker. header.stamp = stamp
    bb_marker.id = id
    bb_marker.type = Marker.LINE_LIST
    bb_marker.action = Marker.ADD
    bb_marker.scale.x = bb_marker.scale.y = bb_marker.scale.z = 0.05
    c = ColorRGBA()
    c.a = 1.0
    c.r = 1.0
    c.g = 0.0
    c.b = 0.0
    bb_marker.pose.orientation.x = 0.0
    bb_marker.pose.orientation.y = 0.0
    bb_marker.pose.orientation.z = 0.0
    bb_marker.pose.orientation.w = 1.0
    bb_marker.pose.position.x = 0.0
    bb_marker.pose.position.y = 0.0
    bb_marker.pose.position.z = 0.2

    for box in b_boxes:
        # Vertexes of the rectangle
        p1 = Point()
        p1.x = box[0][0]
        p1.y = box[0][1]
        p2 = Point()
        p2.x = box[1][0]
        p2.y = box[1][1]
        p3 = Point()
        p3.x = box[2][0]
        p3.y = box[2][1]
        p4 = Point()
        p4.x = box[3][0]
        p4.y = box[3][1]
        p1.z = 0.1
        p2.z = 0.1
        p3.z = 0.1
        p4.z = 0.1
        # Compose lines
        bb_marker.points.append(p1)
        bb_marker.points.append(p2)
        bb_marker.points.append(p2)
        bb_marker.points.append(p3)
        bb_marker.points.append(p3)
        bb_marker.points.append(p4)
        bb_marker.points.append(p4)
        bb_marker.points.append(p1)
        bb_marker.colors.append(c)
    bb_marker.color = c
    bbox_marker_pub.publish(bb_marker)