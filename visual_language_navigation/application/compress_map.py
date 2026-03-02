# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Leonardo Gori

from visual_language_navigation.map.voxel_feat_map import VoxelFeatMap
import numpy as np
import argparse
import psutil

# Use this utility to optimize the disk space of maps recorded before the memory saving update
# python compress_map.py /path/to/source_map.h5df /path/to/destination_map.h5df

def main() -> None:
    # Set up argument parser to take map file paths as inputs
    parser = argparse.ArgumentParser(description="Process voxel maps.")
    parser.add_argument("map_path", type=str, help="Path to the source map")
    parser.add_argument("dest_map_path", type=str, help="Path to save the processed map")
    
    args = parser.parse_args()
    
    map_path = args.map_path
    dest_map_path = args.dest_map_path

    print(f"Memory information before loading map: {psutil.virtual_memory()}")

    voxel_map = VoxelFeatMap(save_memory_at_runtime=True)

    used_memory_before_map_loading = psutil.virtual_memory()[3]/1000000000
    if not voxel_map.load_map(map_path):
        print(f"Unable to load map from: {map_path}")
        return

    print(f"Memory information after loading map: {psutil.virtual_memory()}")

    print(f"memory loaded for the map: {psutil.virtual_memory()[3]/1000000000} - {used_memory_before_map_loading} = {psutil.virtual_memory()[3]/1000000000 - used_memory_before_map_loading}")
    
    embs, counts = np.unique(voxel_map.grid_feat, return_counts=True, axis=0)
    print(f"Distict embeddings shape is: {embs.shape}, meaning that there are {embs.shape[0]} embeddings of size {embs.shape[1]} over a total of {counts.sum()} voxel embeddings.")

    voxel_map.save_map("/home/user1/test.h5df")
    print(f"Processed map saved to: {dest_map_path}")
    
if __name__ == "__main__":
    main()