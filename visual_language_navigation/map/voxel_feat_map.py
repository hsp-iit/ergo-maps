# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from visual_language_navigation.utils.mapping_utils import load_3d_map, save_3d_map, to_grid_coords
from visual_language_navigation.utils.clip_utils import get_lseg_score
import threading
import torch
import torch.nn.functional
from transformers import CLIPTokenizer, CLIPModel


class VoxelFeatMap:
    def __init__(self, 
                 seg_model,
                 map_name = "voxel_feat_map",
                 grid_size = 1000,
                 cell_size = 0.05,
                 save_memory_at_runtime : bool = False,
                 seg_model_name = "fcclip"
                 ):
        # Name of the map
        self.map_name = map_name
        # N x N size of the voxel grid, N is the number of voxels
        self.grid_size = grid_size
        # Cell/voxel size of the map, in meters
        self.cell_size = cell_size
        # CUDA
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        # wether to apply the saving memory approach or not
        self.save_memory_at_runtime = save_memory_at_runtime
        # Sanity check
        if seg_model_name == "lseg" and seg_model_name != "fcclip":
            self.seg_model_name = "lseg"
            model_name = 'openai/clip-vit-base-patch32'
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.torch_device)
        # FCCLIP
        elif seg_model_name != "lseg" and seg_model_name == "fcclip":
            self.seg_model_name = seg_model_name
        else:   
            print(f"[VoxelFeatMap:__init__] ERROR: found unsupported segmentation model: {seg_model_name=}. Using default fcclip")
            self.seg_model_name = "fcclip"    #default
        
        # Used only by FCCLIP
        self.clip_model = seg_model
        # Grid values of:
        # list of distinct Embeddings/clip features ordered by id
        self.grid_feat = None
        # list of occupied ids ordered by id
        self.grid_pos = None
        # list of each voxel weight (from feature fusion) ordered by id
        self.weight = None
        # Grid containing the ID of the occupied voxels, if -1 is empty, 0 has been cleared, >= 1 it's the ID
        self.occupied_ids = None
        # Grid containing the RGB value of each voxel
        self.grid_rgb = None
        # Integer value of the maximum voxel ID mapped
        self.max_id = None
        # Voxel grid maximum height
        self.grid_height = None   # Needs to be set in init_map()
        # List of flags to enable/disable voxels
        self.voxels_flags = None
        self.data_lock = threading.Lock()
        self.map_present = False

        # dict where the key is the bytecode of the np array representing the embedding and the value is the index that we associate to that embedding
        # the index will be present in self.grid_feat, which maps to this dict to save memory
        # this optimizes the pipeline at runtime since the average access complexity to the key of a python dict is O(1)
        self.grid_feature_index_dict = {}
        # the counter allows to keep track of the index of the distinct embeddings seen at runtime
        # list of distinct Embeddings/clip features (vector of 512 floats) ordered by id
        self.grid_feat_index = None
        self.grid_feature_counter = 1 # start from 1 because 0 is used in remove_map_voxels to clear voxels
        # distinct feat array starts with the zero embedding, which is a fake embedding for voxels that have been erased from the map
        # add a fake embedding in position zero, which represents the zero embedding, i.e. the voxel associated to this id have been erased
        self.distinct_feat_array = torch.zeros(1, 768 + 768)

    # function that loads in memory the grid features from the dict and index list, that may appear multiple times
    # TODO: check if it's better to use the dict and update it at runtime, or to produce it only when the map is actually saved on disk
    def _preload_grid_feat(self, grid_feat_index, distinct_feat_array):
        feat_dim = distinct_feat_array.shape[-1]
        num_voxels = grid_feat_index.shape[0]

        self.grid_feat = np.zeros((num_voxels, feat_dim), dtype=np.float32)

        for i, feat_id in enumerate(grid_feat_index):
            feat_id = int(feat_id.item())
            # if the index of the feature is 0, then it means that this voxel has been removed, therefore keep the zeros numpy array
            if feat_id == 0:
                continue
            # otherwise assign to the grid feat the embedding associated to the feat_id
            self.grid_feat[i] = distinct_feat_array[feat_id]

    def init_map(self, maximum_height: float, map_path: Path, clip_feat_dim : int) -> bool:
        """
        Initialize a squared voxel grid of size (gs, gs, gh), gh = maximum_height / cs, each voxel is of
        size cs
        Call this method if you need to create a brand new map, or change the maximum height parameter

        :param maximum_height: the maximum height for the map, in meters
        :param map_path: the absolute path of the map. Will attempt to load it, otherwise will create an empty one and save there.
        :param clip_feat_dim: size of the feature vector

        :return: True or False whether an already existing map has been found and loaded
        """
        # check if there is already saved map
        if self.load_map(map_path):
            loaded_map = True
            self.grid_height = int(maximum_height / self.cell_size)
        else:
            with self.data_lock:
                # init the map related variables 
                self.grid_height = int(maximum_height / self.cell_size)
                # 
                if self.save_memory_at_runtime:
                    self.grid_feat_index = np.zeros((self.grid_size * self.grid_size, 1), dtype=np.int32)
                else:
                    self.grid_feat = np.zeros((self.grid_size * self.grid_size, clip_feat_dim), dtype=np.float32)
                self.grid_pos = np.zeros((self.grid_size * self.grid_size, 3), dtype=np.int32)
                self.occupied_ids = -1 * np.ones((self.grid_size, self.grid_size, self.grid_height), dtype=np.int32)
                self.voxels_flags = np.zeros((self.grid_size * self.grid_size), dtype=np.uint8)
                self.weight = np.zeros((self.grid_size * self.grid_size), dtype=np.float32)
                self.grid_rgb = np.zeros((self.grid_size * self.grid_size, 3), dtype=np.uint8)
                self.max_id = 0
                loaded_map = False
                self.map_present = False
        return loaded_map

    def load_map(self, map_path: str) -> bool:
        """
        Load a voxel grid of size (gs, gs, gh), gh = maximum_height / cell_size, 
        each voxel is of size cs
        """
        with self.data_lock:
            try:
                if os.path.exists(map_path):
                    (
                        grid_feat_data,
                        self.grid_pos,
                        self.weight,
                        self.occupied_ids,
                        self.voxels_flags,
                        self.grid_rgb,
                        self.cell_size,
                        self.grid_size
                    ) = load_3d_map(map_path)

                    if "grid_feat" not in grid_feat_data:
                        self.max_id = grid_feat_data["grid_feat_index"].shape[0] - 1
                        if self.save_memory_at_runtime:
                            self.distinct_feat_array = torch.tensor(grid_feat_data["distinct_feat_array"], dtype=torch.float32)
                            self.grid_feat_index = grid_feat_data["grid_feat_index"].astype(np.int32)
                            self.grid_feature_counter = self.distinct_feat_array.shape[0]
                            for i, feat in enumerate(self.distinct_feat_array):
                                self.grid_feature_index_dict[np.array(feat).tobytes()] = i
                        else:
                            self._preload_grid_feat(grid_feat_data["grid_feat_index"], grid_feat_data["distinct_feat_array"])
                    else:
                        self.max_id = grid_feat_data["grid_feat"].shape[0] - 1
                        self.grid_feat = grid_feat_data["grid_feat"]
                    
                    print(f"[VoxelFeatMap:load_map] Loaded map from: {map_path}")
                    self.map_present = True
                    return True
                else:
                    print(f"[VoxelFeatMap:load_map] Invalid Path: {map_path}")
                    return False
            except Exception as ex:
                print(f"[VoxelFeatMap:load_map] An exception occurred: {ex} while loading {map_path=}")
                return False
    
    def save_map(self, save_path):
        """
        save a voxel grid of size up to max_id. Each voxel is of size cell_size
        """
        with self.data_lock:
            is_feat_data_shape_None = False
            if self.save_memory_at_runtime:
                is_feat_data_shape_None = self.grid_feat_index.shape is None
            else:
                is_feat_data_shape_None = self.grid_feat.shape is None
            
            if self.grid_pos is None or is_feat_data_shape_None or self.occupied_ids is None:
                print("[VoxelFeatMap:save_map] Called save_map with empty map: not saving")
            else:
                if self.save_memory_at_runtime:
                    grid_feat_data = {"grid_feat_index": self.grid_feat_index, "grid_feat_index_dict" : self.grid_feature_index_dict}
                else:
                    grid_feat_data = {"grid_feat": self.grid_feat}
                grid_feat_data["save_memory_at_runtime"] = self.save_memory_at_runtime

                save_3d_map(save_path, grid_feat_data, self.grid_pos, self.weight, self.occupied_ids, self.voxels_flags, self.grid_rgb, self.max_id, self.cell_size, self.grid_size)

    def index_map_lseg(self, language_desc: str, 
                        clip_model, 
                        lseg_clip_feat_dim = 512, 
                        lseg_use_multiple_templates = True,
                        lseg_add_other = True, 
                        lseg_avg_mode = 0, 
                        lseg_use_cosine_sim=True, 
                        lseg_clip_threshold = 0.90,
                        batch_size = 2**14):
        """
        Indexes a voxel grid map based on semantic similarity between stored voxel embeddings 
        and a given language description, producing a boolean mask of relevant voxels.
    
        This function computes the similarity between each voxel's feature embedding 
        (stored in `self.grid_feat`) and the CLIP embedding of the input `language_desc`.
        Depending on the configuration, it either uses cosine similarity directly 
        or a precomputed LSeg scoring method. Voxels exceeding a similarity threshold 
        are marked as matching the input query.
    
        Parameters
        ----------
        language_desc : str
            The natural language description or query to locate in the map.
        clip_model : Any
            The CLIP model or identifier used for text and image embedding comparison.
        lseg_clip_feat_dim : int, optional, default=512
            Dimensionality of the CLIP feature embeddings.
        lseg_use_multiple_templates : bool, optional, default=True
            Whether to use multiple textual templates for the query when computing LSeg scores.
        lseg_add_other : bool, optional, default=True
            Whether to include an "other" category when computing LSeg scores for contrast.
        lseg_avg_mode : int, optional, default=0
            Averaging mode used in the LSeg scoring process (implementation-dependent).
        lseg_use_cosine_sim : bool, optional, default=True
            If True, uses cosine similarity directly between voxel and text embeddings.
            If False, uses the LSeg score computation function instead.
        lseg_clip_threshold : float, optional, default=0.90
            Cosine similarity threshold above which a voxel is considered a match.
        batch_size : int, optional, default=2**14
            Number of voxel features processed per batch for memory efficiency.
    
        Returns
        -------
        numpy.ndarray
            A boolean mask of shape `(N,)` where each element indicates whether 
            the corresponding voxel matches the language query (`True`) or not (`False`).
    
        """
        if lseg_use_cosine_sim:
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    language_desc, 
                    padding="max_length", 
                    return_tensors="pt",
                    ).to(self.torch_device)
                text_feature = self.model.get_text_features(**text_inputs)
                #batch_size = 4096
                masks = []
                self.data_lock.acquire()
                # Use batches for efficient memory usage
                for i in range(0, self.grid_feat.shape[0], batch_size):
                    torch_grid = torch.tensor(self.grid_feat[i: i + batch_size]).to(self.torch_device)
                    distances = torch.nn.functional.cosine_similarity(torch_grid, text_feature, dim=1)
                    masks.append(distances.cpu().detach().numpy() > lseg_clip_threshold)
                self.data_lock.release()
                mask = np.concatenate(masks) 
        else:
            self.data_lock.acquire()
            scores_mat = get_lseg_score(
                clip_model,
                [language_desc],
                self.grid_feat,
                lseg_clip_feat_dim,
                use_multiple_templates=lseg_use_multiple_templates,
                add_other=lseg_add_other,
                avg_mode=lseg_avg_mode
            )  # score for the query text and 'other'
            self.data_lock.release()
            cat_id = 0
            max_ids = np.argmax(scores_mat, axis=1) # selects which one is closer, is 0 if closer to the query, 1 otherwise
            mask = max_ids == cat_id
        return mask
    
    def index_map_fcclip(self, language_desc: str,
                        fcclip_m2f_cos_sim_threshold = 0.22, 
                        fcclip_clip_cos_sim_threshold = 0.25, 
                        batch_size = 2**14):
        """
        Indexes a voxel grid map using the FC-CLIP model to identify regions matching
        a given natural language description.

        This function computes voxel-wise similarity scores between the map embeddings 
        and the CLIP embeddings of the input `language_desc`. It uses the FC-CLIP 
        segmentation model to evaluate semantic alignment based on multimodal 
        cosine similarity thresholds. Voxels with scores exceeding the specified thresholds are marked as matches.

        Parameters
        ----------
        language_desc : str
            The natural language description or query to locate in the map.
        fcclip_m2f_cos_sim_threshold : float
            Cosine similarity threshold for mask2former (M2F) component.
            Higher values make matching more selective.
        fcclip_clip_cos_sim_threshold : float
            Cosine similarity threshold for the CLIP feature matching component.
        batch_size : int, optional, default=2**14
            Number of voxel features processed per batch for efficient GPU memory usage.

        Returns
        -------
        numpy.ndarray
            A boolean mask of shape `(N,)` where `True` indicates that the voxel 
            semantically matches the `language_desc` query.
        """
        #FCCLIP
        if self.seg_model_name == "fcclip":
            all_scores = []
            self.data_lock.acquire()
            with torch.no_grad():
                if self.save_memory_at_runtime:
                    map_embeddings = self.distinct_feat_array
                else:
                    map_embeddings = self.grid_feat
                for start_idx in range(0, map_embeddings.shape[0], batch_size):
                    end_idx = start_idx + batch_size
                    batch_feats = map_embeddings[start_idx:end_idx]  # Slice the current batch

                    # Convert to torch tensor and add batch dimension
                    if isinstance(map_embeddings, torch.Tensor):
                        map_feats = batch_feats.detach().unsqueeze(0).cuda()
                    else:
                        map_feats = torch.tensor(batch_feats).unsqueeze(0).cuda()

                    # Get segmentation scores for the current batch
                    with torch.no_grad():  # Disable gradients to save memory
                        batch_scores = self.clip_model.query_segmentation([[language_desc]], 
                                                                          map_feats, 
                                                                          m2f_cos_sim_threshold = fcclip_m2f_cos_sim_threshold, 
                                                                          clip_cos_sim_threshold = fcclip_clip_cos_sim_threshold)
                    # convert 1 to 0 and 0 to 1 to match boolean values
                    batch_scores = 1 - batch_scores
                    # Append to list and move data back to CPU
                    all_scores.append(batch_scores.squeeze(0).cpu().numpy())

            # Concatenate the batch results along the first dimension
            scores_list = np.concatenate(all_scores, axis=0)
            if self.save_memory_at_runtime:
                # remap the mask list to the voxel list
                res_mask_list = scores_list[self.grid_feat_index.astype(int)].astype(bool).flatten()
            else:
                res_mask_list = scores_list.astype(bool)
            self.data_lock.release()
            return res_mask_list
    
    def update_map(self, points_xyz, embeddings, rgb, category_preds, inds_to_remove, use_feature_fusion : bool, cam_pose):
        """
        Updates the voxel grid map with new point cloud data, color, embeddings semantic information.

        This function integrates 3D points and their corresponding feature embeddings into 
        the voxelized map representation. It supports both direct feature replacement 
        and feature fusion (weighted averaging) based on spatial distance to the camera. 
        It can also operate in memory-efficient mode by storing distinct embeddings and 
        referencing them via indices.

        Parameters
        ----------
        points_xyz : numpy.ndarray or torch.Tensor
            Array of 3D point coordinates in world or camera space, shape `(N, 3)`.
        embeddings : numpy.ndarray
            Aligned feature embeddings for each point, shape `(N, D)`, with `points_xyz`.
        rgb : numpy.ndarray
            Array of RGB color values for each point, shape `(N, 3)`, aligned with `points_xyz`.
        category_preds : numpy.ndarray
            Predicted category indices for each point, shape `(N,)`.
        inds_to_remove : list or numpy.ndarray
            List of category indices that should be ignored (e.g., invalid or background).
        use_feature_fusion : bool
            If True, applies distance-weighted feature fusion per ConceptFusion 
            (https://arxiv.org/pdf/2302.07241.pdf, Sec. 4.1). If False, overwrites features directly.
        cam_pose : numpy.ndarray
            Camera position used to compute distance-based weighting for feature fusion, shape `(3,)`.
    """
        self.data_lock.acquire()
        # if embeddings is a tuple, then we passed distinct embeddings and an index map with size of the image
        mask_embeddings = None
        if isinstance(embeddings, tuple):
            # first argument is the mask embedding list
            # second argument is the index map with size of the image
            mask_embeddings, embeddings = embeddings[0], embeddings[1]
        for (point, feature, rgb, category_pred) in zip(points_xyz, embeddings, rgb, category_preds):
            # If it's a -1 value, this means that the associated embedding is the zero embedding, therefore skip it
            if self.seg_model_name == "fcclip" and np.sum(feature) == -1:
                continue
            
            row, col, height = to_grid_coords(point[0], point[1], point[2], self.grid_size, self.cell_size)
            if self.out_of_range(row, col, height) or (category_pred in inds_to_remove):
                continue

            # when the max_id (number of points) exceeds the reserved maximum size,
            # double the grid_feat, grid_pos, weight, grid_rgb lengths
            if self.save_memory_at_runtime:
                if self.max_id >= self.grid_feat_index.shape[0]:
                    self.reserve_map_space()
            else:
                if self.max_id >= self.grid_feat.shape[0]:
                    self.reserve_map_space()

            # apply the distance weighting according to
            # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
            radial_dist_sq = np.sum(np.square(point - cam_pose))
            sigma_sq = 0.36  #TODO parameterize
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            # if mask embeddings is not None, then we passed the tuple (mask embeddings, index map)
            if mask_embeddings is not None:
                feat = mask_embeddings[feature.astype(int)][0]
            else:
                feat = feature

            occupied_id = self.occupied_ids[row, col, height]

            if use_feature_fusion:
                # implementing the saving memory approach for feature fusion option, but in theory it's not effective
                # TODO: discuss about removing this option for FC-CLIP or generic 2-stage segmentation models
                if occupied_id == -1:
                    self.occupied_ids[row, col, height] = self.max_id
                    weigthed_emb = feat * alpha
                    # if it's the first time you meet the weighted embedding, save it into the distinct embedding dict, and associate it with a new counter value
                    # check if weighted_emb in dict in O(1) average

                    if self.save_memory_at_runtime:
                        if weigthed_emb.tobytes() not in self.grid_feature_index_dict:
                            self.grid_feature_index_dict[weigthed_emb.tobytes()] = self.grid_feature_counter
                            self.grid_feature_counter += 1

                        # wether or not it's the first time you meet the weighted embedding, assign to the grid feat the index associated to it
                        self.grid_feat_index[self.max_id] = self.grid_feature_index_dict[weigthed_emb.tobytes()]
                    else:
                        self.grid_feat[self.max_id] = weigthed_emb
                    
                    self.grid_rgb[self.max_id] = rgb
                    self.weight[self.max_id] += alpha
                    self.grid_pos[self.max_id] = [row, col, height]
                    self.voxels_flags[self.max_id] = 1
                    self.max_id = self.max_id + 1
                else:
                    if self.save_memory_at_runtime:
                        weigthed_emb = (
                            self.grid_feat_index[occupied_id] * self.weight[occupied_id] + feat * alpha
                        ) / (self.weight[occupied_id] + alpha)
                        # if it's the first time you meet the weighted embedding, save it into the distinct embedding dict, and associate it with a new counter value
                        # check if weighted_emb in dict in O(1) average
                        if weigthed_emb.tobytes() not in self.grid_feature_index_dict:
                            self.grid_feature_index_dict[weigthed_emb.tobytes()] = self.grid_feature_counter
                            self.grid_feature_counter +=1
                        self.grid_feat_index[occupied_id] = self.grid_feature_index_dict[weigthed_emb.tobytes()]
                    else:
                        self.grid_feat[occupied_id] = (self.grid_feat[occupied_id] * self.weight[occupied_id] + feat * alpha) / (self.weight[occupied_id] + alpha)

                    self.grid_rgb[occupied_id] = rgb
                    self.weight[occupied_id] += alpha
                    self.voxels_flags[occupied_id] = 1
                    
            else:
                if self.seg_model_name == "fcclip":
                    assert feat.shape[0] == 1536

                if occupied_id == -1:
                    self.occupied_ids[row, col, height] = self.max_id

                    if self.save_memory_at_runtime:
                        # if it's the first time you meet feat, save it into the distinct embedding dict, and associate it with a new counter value
                        # check if feat in dict in O(1) average
                        if feat.tobytes() not in self.grid_feature_index_dict:
                            self.grid_feature_index_dict[feat.tobytes()] = self.grid_feature_counter
                            self.distinct_feat_array = torch.cat((self.distinct_feat_array, torch.tensor(feat).unsqueeze(0)))
                            assert self.distinct_feat_array[self.grid_feature_counter].equal(torch.tensor(feat))
                            self.grid_feature_counter +=1

                        self.grid_feat_index[self.max_id] = self.grid_feature_index_dict[feat.tobytes()]
                    else:
                        self.grid_feat[self.max_id] = feat

                    self.grid_rgb[self.max_id] = rgb
                    self.weight[self.max_id] += alpha
                    self.grid_pos[self.max_id] = [row, col, height]
                    self.voxels_flags[self.max_id] = 1
                    self.max_id = self.max_id + 1
                else:
                    if self.save_memory_at_runtime:
                        # if it's the first time you meet feat, save it into the distinct embedding dict, and associate it with a new counter value
                        # check if feat in dict in O(1) average
                        if feat.tobytes() not in self.grid_feature_index_dict:
                            self.grid_feature_index_dict[feat.tobytes()] = self.grid_feature_counter
                            self.distinct_feat_array = torch.cat((self.distinct_feat_array, torch.tensor(feat).unsqueeze(0)))
                            assert self.distinct_feat_array[self.grid_feature_counter].equal(torch.tensor(feat))
                            self.grid_feature_counter +=1

                        self.grid_feat_index[occupied_id] = self.grid_feature_index_dict[feat.tobytes()]
                    else:
                        self.grid_feat[occupied_id] = feat

                    self.grid_rgb[occupied_id] = rgb
                    self.weight[occupied_id] += alpha
                    self.voxels_flags[occupied_id] = 1
        self.map_present = True  #flag that map is present
        self.data_lock.release()
    
    def remove_map_voxels(self, voxels_to_clear):
        """
        :param voxels_to_clear: array of shape (N, 3) with the voxels in the global grid map frame to be removed from self.occupied_ids
        :return: True or False upon successful completion
        """
        self.data_lock.acquire()
        try:
            if voxels_to_clear.size != 0:
                for voxel in voxels_to_clear:
                    # Check if in range
                    if self.out_of_range(voxel[0], voxel[1], voxel[2]):
                        continue
                    # Check if voxel is already mapped and visible:
                    if (int(self.occupied_ids[voxel[0], voxel[1], voxel[2]]) > -1) and (int(self.voxels_flags[self.occupied_ids[voxel[0], voxel[1], voxel[2]]]) != 0):
                        self.voxels_flags[self.occupied_ids[voxel[0], voxel[1], voxel[2]]] = 0     # 0 means ignore
            self.data_lock.release()
            return True
        except Exception as ex:
            print(f"[VoxelFeatMap:remove_map_voxels] An Exception occurred: {ex}")
            self.data_lock.release()
            return False
        
        
    def out_of_range(self, row: int, col: int, height: int) -> bool:
        return col >= self.grid_size or row >= self.grid_size or height >= self.grid_height or col < 0 or row < 0 or height < 0
    
    def reserve_map_space(self):
        if self.save_memory_at_runtime:
            self.grid_feat_index = np.concatenate(
                [
                    self.grid_feat_index,
                    np.zeros((self.grid_feat_index.shape[0], self.grid_feat_index.shape[1]), dtype=np.float32),
                ],
                axis=0,
            )
        else:
            self.grid_feat = np.concatenate(
                [
                    self.grid_feat,
                    np.zeros((self.grid_feat.shape[0], self.grid_feat.shape[1]), dtype=np.float32),
                ],
                axis=0,
            )
        self.grid_pos = np.concatenate(
            [
                self.grid_pos,
                np.zeros((self.grid_pos.shape[0], self.grid_pos.shape[1]), dtype=np.int32),
            ],
            axis=0,
        )
        self.weight = np.concatenate([self.weight, np.zeros((self.weight.shape[0]), dtype=np.int32)], axis=0)
        self.grid_rgb = np.concatenate(
            [
                self.grid_rgb,
                np.zeros((self.grid_rgb.shape[0], self.grid_rgb.shape[1]), dtype=np.float32),
            ],
            axis=0,
        )
        self.voxels_flags = np.concatenate([self.voxels_flags, np.zeros((self.voxels_flags.shape[0]), dtype=np.uintc)], axis=0)
