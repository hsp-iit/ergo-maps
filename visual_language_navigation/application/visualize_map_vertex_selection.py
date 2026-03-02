from visual_language_navigation.utils.mapping_utils import load_3d_map
from visual_language_navigation.utils.visualize_utils import visualize_rgb_map_3d
import os

def main(args = None):
    map_name = "floor0_noMap012_2mm_3max"
    map_path = f"/home/user1/vlmaps_files/{map_name}.h5df"
    try:
        if os.path.exists(map_path):
            (
                _,
                grid_pos,
                _,
                occupied_ids,
                voxels_flags,
                grid_rgb,
                cs,
                _
            ) = load_3d_map(map_path)
        visible_voxels = grid_pos[voxels_flags == 1]
        visible_colors = grid_rgb[voxels_flags == 1]
        visualize_rgb_map_3d(visible_voxels, visible_colors, "vis")

    except Exception as ex:
        print(f"{ex=}")
        return


if __name__ == "__main__":
    main()