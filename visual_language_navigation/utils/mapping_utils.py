# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import numpy as np
import torch
import h5py
from typing import Tuple
from collections import deque
from tf2_ros import StaticTransformBroadcaster, TransformStamped
import visual_language_navigation.utils.traverse_pixels as raycast

def convert_pc_to_grid(grid_size, cell_size, pointcloud):
    """
    Converts a pointcloud tensor, in map frame, to the semantic map grid frame
    """
    grid = torch.zeros_like(pointcloud)
    grid[:, :2] = (grid_size / 2 - (pointcloud[:, :2] / cell_size).to(torch.int))
    grid[:, 2] = (pointcloud[:, 2] / cell_size)
    return grid.to(torch.float32)

def to_grid_coords(x, y, z, grid_size, cell_size):
    """
    Coverts the coordinates from meters into grid voxels units
    """
    row = int(grid_size / 2 - int(x / cell_size))
    col = int(grid_size / 2 - int(y / cell_size))
    h = int(z / cell_size)
    return [row, col, h]
    
def from_grid_coords(row, col, height, grid_size, cell_size):
    """
    Converts the coordinates of a voxel in meters
    """
    x = (grid_size / 2 - row) * cell_size
    y = (grid_size / 2 - col) * cell_size
    z = height * cell_size
    return [x, y, z]

def publish_static_transform(stamp, frame_id, child_frame_id, grid_size, cell_size, tf_static_broadcaster : StaticTransformBroadcaster):
    """
    Publish a static transform between the global map frame and the voxel map frame.

    Args:
        stamp (builtin_interfaces.msg.Time):
            ROS2 timestamp to assign to the transform header.
        frame_id (str):
            Name of the parent frame (typically the global map frame).
        child_frame_id (str):
            Name of the child frame (typically the voxel map frame).
        grid_size (int):
            Number of voxels along each axis of the grid GxG.
        cell_size (float):
            Size of each voxel cell in meters.
        tf_static_broadcaster (tf2_ros.StaticTransformBroadcaster):
            ROS2 broadcaster object used to send the static transform.

    Returns:
        bool:
            True if the transform was successfully published, False otherwise.
    """

    try:
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = frame_id           #self.target_frame
        tf.child_frame_id = child_frame_id      #self.semantic_map_frame_name
        tf.transform.translation.x = (grid_size * cell_size) / 2
        tf.transform.translation.y = (grid_size * cell_size) / 2
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = -1.0
        tf.transform.rotation.w = 0.0
        tf_static_broadcaster.sendTransform(tf)
        return True
    except Exception as ex:
        print(f"{ex=}")
        return False

def raycasting(camera_pose: np.ndarray, 
               camera_cloud: torch.Tensor, 
               grid_size, 
               cell_size, 
               grid_cloud, 
               voxels_flags, 
               raycast_distance_threshold = 0.5, 
               voxel_offset = 0, 
               camera_batch_sz = 2**11, 
               map_batch_sz = 2**10, 
               device = "cuda"):
    """
    Traces a bundle of lines from the camera pose to each point in the camera pointcloud.
    Checks if any voxel of the map is close enough to each ray, if so that voxel will be listed for removal.

    For each point in the camera's depth point cloud, a ray is traced from the camera pose 
    to that point. Map voxels that lie sufficiently close to one of these rays are flagged 
    for removal if they fall within the bounding region of the camera's current field of view.

    Args:
        camera_pose (np.ndarray): 
            The camera position expressed in map frame coordinates, shape (1, 3).
        camera_cloud (torch.Tensor): 
            Camera point cloud expressed in map frame coordinates, shape (N, 3).
        grid_size (int): 
            Dimensions of the voxel grid GxG.
        cell_size (float): 
            Size of each voxel cell in meters.
        grid_cloud (np.ndarray or torch.Tensor): 
            Global voxel grid coordinates of shape (M, 3).
        voxels_flags (np.ndarray): 
            Occupancy flags for each voxel (nonzero = active/occupied).
        raycast_distance_threshold (float): 
            Maximum distance in voxel units between a ray and a voxel center 
            for the voxel to be considered hit. Default is 0.5.
        voxel_offset (int): 
            Optional integer offset applied to ray-voxel intersections, 
            used to dilate or shrink the cleared region. Default is 0.
        camera_batch_sz (int): 
            Number of camera points to process per batch for memory efficiency. Default is 2**11.
        map_batch_sz (int): 
            Number of voxels to process per batch during raycasting. Default is 2**10.
        device (str): 
            Torch device to run computations on ("cuda" or "cpu"). Default is "cuda".

    Returns:
        np.ndarray: 
            Array of voxel indices (shape (K, 3)) to clear from the map, 
            where K is the number of voxels identified as free space.
    """
    # Pass to tensors
    torch.cuda.empty_cache()
    camera_pose_voxel = to_grid_coords(camera_pose[0], camera_pose[1], camera_pose[2], grid_size, cell_size)
    camera_pose_voxel_torch = torch.tensor(camera_pose_voxel, device=device, dtype=torch.float32)
    cam_voxels_torch = convert_pc_to_grid(grid_size, cell_size, camera_cloud)

    # Return data
    voxels_to_clear_list = []
    # Filter points on the map only in the bounding box of the camera pointcloud + camera)
    map_points = torch.tensor(grid_cloud, device=device, dtype=torch.float32)
    cam_points_w_cam = torch.cat([cam_voxels_torch, camera_pose_voxel_torch.unsqueeze(0)], dim=0)
    points_min = torch.min(cam_points_w_cam, dim=0)[0]
    points_max = torch.max(cam_points_w_cam, dim=0)[0]
    # Consider only the active ones
    visible_voxels = torch.tensor((voxels_flags > 0), device=device, dtype=torch.bool)
    mask = torch.all((map_points >= points_min) & (map_points <= points_max), dim=1)
    map_points = map_points[mask & visible_voxels]
    # We add the camera point to each batch to determine the bounding box (not optimal, should take a frustum)
    for i in range(0, cam_voxels_torch.shape[0], camera_batch_sz):
        pc_end = min(i + camera_batch_sz, cam_voxels_torch.shape[0])
        batch_cloud = torch.cat([cam_voxels_torch[i:pc_end], camera_pose_voxel_torch.unsqueeze(0)], dim=0)
        min_bounds = torch.min(batch_cloud, dim=0)[0]  # Minimum x, y, z of the camera pointcloud
        max_bounds = torch.max(batch_cloud, dim=0)[0]  # Maximum x, y, z of the camera pointcloud
        map_mask = torch.all((map_points >= min_bounds) & (map_points <= max_bounds), dim=1)
        map_points_batch = map_points[map_mask]
            
        voxels_to_clear = (raycast.raycast(camera_pose_voxel_torch,
                                           cam_voxels_torch[i:pc_end],
                                           map_points_batch,
                                           batch_size= map_batch_sz,
                                           distance_threshold = raycast_distance_threshold,
                                           offset=voxel_offset)).to(torch.int)
        voxels_to_clear = voxels_to_clear.detach().cpu().numpy()
        voxels_to_clear_list.extend(voxels_to_clear)
    # Free GPU memory
    del batch_cloud, cam_voxels_torch, camera_pose_voxel_torch, map_points, map_points_batch, map_mask, max_bounds, min_bounds
    return np.array(voxels_to_clear_list)

# The grid is a boolean matrix where true = occupied and false = free
def find_closest_free_cell(grid, start_row, start_col):
    width, height = len(grid), len(grid[0])
    if not grid[start_row][start_col]:  # If the starting cell is already free, return it
        return start_row, start_col

    # Initialize BFS
    queue = deque([(start_row, start_col)])
    visited = set((start_row, start_col))
    # Directions for moving in the grid: right, down, left, up
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        current_row, current_col = queue.popleft()

        # Check all neighboring cells
        for dr, dc in directions:
            new_row, new_col = current_row + dr, current_col + dc
            # Ensure new cell is within grid bounds and hasn't been visited
            if 0 <= new_row < width and 0 <= new_col < height and (new_row, new_col) not in visited:
                if not grid[new_row][new_col]:  # If the cell is free
                    return new_row, new_col

                # Mark as visited and add to queue if occupied
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))

    return None  # Return None if no free cell is found

def save_3d_map(
    save_path: str,
    grid_feat_data: dict,
    grid_pos: np.ndarray,
    weight: np.ndarray,
    occupied_ids: np.ndarray,
    voxels_flags : np.ndarray,
    grid_rgb: np.ndarray = None,
    max_id = None,
    cell_size = 0.05,
    grid_size = 1000
) -> None:
    """Save 3D voxel map with features

    Args:
        save_path (str): path to save the map as an H5DF file.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) the position of the occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        max_id (int, optional): whether to save the arrays up to the max_id mapped, disabled if None
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    if grid_feat_data["save_memory_at_runtime"]:
        grid_feat_index = grid_feat_data["grid_feat_index"]
        grid_feat_index_dict = grid_feat_data["grid_feat_index_dict"]
    else:
        grid_feat = grid_feat_data["grid_feat"]
        zero_embedding = np.zeros_like(grid_feat[0], dtype=np.float32)
        grid_feat_index_dict = {zero_embedding.tobytes() : 0}
        grid_feat_index = np.zeros(grid_feat.shape[0], dtype=np.int32)

        counter = 1
        for i, g_feat in enumerate(grid_feat):
            feat_bytecode = g_feat.tobytes()
            if feat_bytecode not in grid_feat_index_dict:
                grid_feat_index_dict[feat_bytecode] = counter
                counter += 1
            grid_feat_index[i] = grid_feat_index_dict[feat_bytecode]            


    if max_id != None:
        grid_feat_index_ = grid_feat_index[:max_id]
        grid_pos_ = grid_pos[:max_id]
        weight_ = weight[:max_id]
        grid_rgb_ = grid_rgb[:max_id]
        voxels_flags_ = voxels_flags[:max_id]
    else:
        grid_feat_index_ = grid_feat_index
        grid_pos_ = grid_pos
        weight_ = weight
        grid_rgb_ = grid_rgb
        voxels_flags_ = voxels_flags

    # convert the dict to a numpy array
    # the dict contains a number of increasing indices as the number of keys
    grid_index_feat_dict = dict([(idx, np.frombuffer(emb, dtype=np.float32)) for emb, idx in grid_feat_index_dict.items()])

    # add the 0 key which represents the zero embedding, i.e. the voxel associated to this id have been erased
    # TODO: Discuss if this is still necessary
    if 0 not in grid_index_feat_dict:
        grid_index_feat_dict[0] = np.zeros(next(iter(grid_index_feat_dict.values())).shape[-1], dtype=np.float32)

    distinct_feat_array = np.zeros((len(grid_index_feat_dict), next(iter(grid_index_feat_dict.values())).shape[-1]), dtype=np.float32)

    for i in range(len(grid_index_feat_dict)):
        distinct_feat_array[i] = grid_index_feat_dict[i]
    
    with h5py.File(save_path, "w") as f:
        f.create_dataset("grid_feat_index", data=grid_feat_index_)
        f.create_dataset("distinct_feat_array", data=distinct_feat_array)
        f.create_dataset("grid_pos", data=grid_pos_)
        f.create_dataset("weight", data=weight_)
        f.create_dataset("occupied_ids", data=occupied_ids)
        f.create_dataset("voxels_flags", data=voxels_flags_)
        # Save also map parameters
        f.create_dataset("cell_size", data=np.array(cell_size))
        f.create_dataset("grid_size", data=np.array(grid_size))
        if grid_rgb_ is not None:
            f.create_dataset("grid_rgb", data=grid_rgb_)
        print(f"[VoxelFeatMap:save_map] Saved map to: {save_path} with: {grid_feat_index.shape=} {distinct_feat_array.shape=}  {grid_pos_.shape=}  {weight_.shape=}  {occupied_ids.shape=}  {grid_rgb_.shape=}  {max_id=}  {cell_size=}  {grid_size=} {voxels_flags_.shape=}")


def load_3d_map(map_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Load 3D voxel map with features

    Args:
        map_path (str): path to save the map as an H5DF file.
    Return:
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) each row is the (row, col, height) of an occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(map_path, "r") as f:

        if "grid_feat" not in f.keys():
            grid_feat_index = f["grid_feat_index"][:]
            distinct_feat_array = f["distinct_feat_array"][:]
            print(distinct_feat_array.shape)
            print(grid_feat_index.shape)
        else:
            grid_feat = f["grid_feat"][:]
        
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        voxels_flags = f["voxels_flags"][:]
        grid_rgb = None
        if "grid_rgb" in f:
            grid_rgb = f["grid_rgb"][:]
        cell_size = None
        if "cell_size" in f:
            cell_size = float(np.array(f["cell_size"]))
        grid_size = None
        if "grid_size" in f:
            grid_size = int(np.array(f["grid_size"]))

        grid_feat_data = {}
        if "grid_feat" not in f.keys():
            grid_feat_data["grid_feat_index"] = grid_feat_index
            grid_feat_data["distinct_feat_array"] = distinct_feat_array
        else:
            grid_feat_data["grid_feat"] = grid_feat

        
        return grid_feat_data, grid_pos, weight, occupied_ids, voxels_flags, grid_rgb, cell_size, grid_size

