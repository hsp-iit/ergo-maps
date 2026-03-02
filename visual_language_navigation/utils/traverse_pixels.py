# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti
import torch

# Debugger GPU
def print_gpu_usage(debug_print):
    #from pynvml import *
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if (info.used / 1024 ** 2) > 10000.0:
        print("CRITICAL MEMORY LEVEL")
    print(f"{debug_print} Used memory: {info.used / 1024 ** 2:.2f} MB")


def raycast(entry_pos, pc_grid, map, batch_size=2**13, distance_threshold=0.9, offset = 3.0):
    """
    Check if the map voxels are nearby the ray cast from the entry position to the pointcloud.
    
    :param entry_pos: Tensor of shape (1, 3) with the entry position (camera pose) in the map frame
    :param pc_grid: Tensor of shape (N, 3) is the voxelized pointcloud from the camera, in map frame
    :param map: Tensor of shape (M, 3) is the voxelized map
    :param batch_size: (int) the size of the batch used for GPU memory management
    :param distance_threshold: (float) maximum distance allowed from a map voxel to a ray
    :param offset: (float) offset in voxels to stop raycasting before pc_grid is reached
    :return: List of map voxels to be removed by the map
    """
    voxels_to_remove = []
    #crossed_voxels = torch.zeros(map.shape[0], dtype=torch.bool, device="cuda")
    # Do in batches for each portions of the camera cloud
    for i in range(0, pc_grid.shape[0], batch_size):
        # Given the ray r(t) = entry_pos + t x d that goes from the camera to the pointcloud
        # The perpendicular ray passing by the map voxel is:
        # t_p = (map_voxel - entry_pos) x d / norm(d)^2
        # And the projection point to the camera ray (substitution on the first eq):
        # r(t_p) = proj = entry_pos + t_p x d
        # The distance is then: (map_voxel - proj)
        # d = rays
        d = pc_grid[i:i+batch_size] - entry_pos  # directive parameters of the ray: P_pointcloud - P_camera_pose
        d_norm_sq = torch.sum(d**2, dim=1, keepdim=True)  # The squared norm of the directive parameters: for computation efficency later

        # Shapes Conversions for doing in batches
        map_exp = map.unsqueeze(0).expand(d.shape[0], -1, -1)   # (B, M, 3)
        entry_exp = entry_pos.unsqueeze(0).unsqueeze(1)            # (1, 1, 3)
        rays_exp = d.unsqueeze(1)                               # (B, 1, 3)

        # Vector from camera to map points
        v_map_camera = map_exp - entry_exp  # (B, M, 3)

        t = torch.sum(v_map_camera * rays_exp, dim=2) / d_norm_sq  # (B, M)

        proj = entry_exp + t.unsqueeze(-1) * rays_exp  # (B, M, 3)

        # We use the norm for a monodimensional value
        dist = torch.norm(map_exp - proj, dim=2)  # (B, M)

        voxel_offset = offset / torch.sqrt(d_norm_sq)

        # We also need to check t: at t==0 we are at the camera, for t==1 we are at pc
        condition = (dist < distance_threshold) & (t > 0) & (t < (1.0 - voxel_offset))
        #idx = torch.any(condition, dim=0).nonzero(as_tuple=True)[0]
        mask_chunk = map_exp[condition]
        voxels_to_remove.append(mask_chunk)

    if len(voxels_to_remove) > 0:
        voxels_to_remove = torch.unique(torch.cat(voxels_to_remove), dim=0)
    else:
        voxels_to_remove = torch.tensor([], dtype=torch.long, device="cuda")

    return voxels_to_remove

  