from visual_language_navigation.map.voxel_feat_map import VoxelFeatMap
from visual_language_navigation.utils.matterport3d_categories import mp3dcat
from visual_language_navigation.utils.clip_utils import init_clip
from visual_language_navigation.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
)

from visual_language_navigation.utils.fcclip_utils import _init_fcclip

def main() -> None:
    # Change these parameters to play with different view and maps
    seg_model = "fcclip"  #lseg or fcclip
    index_2d = False
    map_path = "/home/user1/semantic_map.h5df"
    decay_rate = 0.01
    save_memory_at_runtime = True

    if seg_model == "lseg":
        feat_model, clip_feat_dim = init_clip()
    elif seg_model == "fcclip":
        feat_model, _ = _init_fcclip()
        clip_feat_dim = 768 + 768

    voxel_map = VoxelFeatMap(save_memory_at_runtime=save_memory_at_runtime, seg_model=feat_model, seg_model_name=seg_model)
    if not voxel_map.load_map(map_path):
        print(f"Unable to load map from: {map_path}")
        return
    visualize_rgb_map_3d(voxel_map.grid_pos, voxel_map.grid_rgb, "semantic_map")
    cat = input("What are you interested in this scene?")

    print("considering categories: ")
    print(mp3dcat[1:-1])
    while True:
        if seg_model == "lseg":
            mask = voxel_map.index_map(cat, feat_model, clip_feat_dim)
        elif seg_model == "fcclip":
            mask = voxel_map.index_map(cat, feat_model)

        if index_2d:
            mask_2d = pool_3d_label_to_2d(mask, voxel_map.grid_pos, voxel_map.grid_size)
            rgb_2d = pool_3d_rgb_to_2d(voxel_map.grid_rgb, voxel_map.grid_pos, voxel_map.grid_size)
            visualize_masked_map_2d(rgb_2d, mask_2d)
            heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=voxel_map.cell_size, decay_rate=decay_rate)
            visualize_heatmap_2d(rgb_2d, heatmap)
        else:
            visualize_masked_map_3d(voxel_map.grid_pos, mask, voxel_map.grid_rgb, "semantic interest")
            
        cat = input("What is your interested category in this scene?")

if __name__ == "__main__":
    main()
