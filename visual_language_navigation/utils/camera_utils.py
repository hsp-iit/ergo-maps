# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import numpy as np
import copy
import torch
import cv2

def project_depth_features_pc_torch(depth : torch.Tensor, features_per_pixels : torch.Tensor, color_img : torch.Tensor, calib_matrix, inds_to_remove = None, category_preds = None, min_depth = 0.2, max_depth = 6.0, depth_factor=1.0, downsampling_factor=10.0, kernel_dilation_size = 20):
        """
        Creates the 3D pointcloud, in camera frame, from the depth and alignes the clip features and RGB color for each 3D point.
        Uses tensors to speed up the process. Uses GPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param features_per_pixels: matrix of shape (1, F, W , H), featured clip embeddings aligned with each RGB pixel. In case of fc-clip seg model, features_per_pixels is just an index image of shape (1, 1, H, W)
        :param color_img: matrix of shape (W , H), color image image from the camera
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :return: numpy array of shape (N, 3) of 3D points, numpy array of shape (N, F) containing aligned CLIP features to each 3D point, numpy array of shape (N, 3) of aligned RGB color for each point
        """
        fx = calib_matrix[0, 0]
        fy = calib_matrix[1, 1]
        cx = calib_matrix[0, 2]
        cy = calib_matrix[1, 2]
        #intrisics = [[fx, 0.0, cx],
        #             [0.0, fy, cy],
        #             [0.0, 0.0, 1.0 / depth_factor]]

        # Hardcoded sanity check
        if min_depth < 0.0:
              min_depth = 0.2
        if max_depth < 0.0:
              max_depth = 6.0

        # Points outside range, will be used for raycasting
        uu_out, vv_out = torch.where(depth >= max_depth)
        depth[uu_out, vv_out] = max_depth   # Cap the depth to avoid raycasting objects over the max depth, to avoid overshooting
        xx_o = (vv_out - cx) * depth[uu_out, vv_out] / fx
        yy_o = (uu_out - cy) * depth[uu_out, vv_out] / fy
        zz_o = depth[uu_out, vv_out] / depth_factor
        pointcloud_out = torch.cat((xx_o.unsqueeze(1), yy_o.unsqueeze(1), zz_o.unsqueeze(1)), 1)

        # filter depth coords based on z distance
        uu, vv = torch.where((depth > min_depth) & (depth < max_depth))
        # Inflate human segmentation mask
        if category_preds is not None:
            binary_mask = np.full_like(category_preds, False)
            infalted_category_preds = category_preds
            if (inds_to_remove is not None) and inds_to_remove!=[]:
                for item in inds_to_remove:
                    binary_mask = binary_mask | (category_preds==item)
                kernel = np.ones((kernel_dilation_size, kernel_dilation_size),np.uint8)
                cv_img = binary_mask.astype(np.uint8)
                infalted_mask = np.array(cv2.dilate(cv_img,kernel,iterations = 1), dtype=bool)
                if len(infalted_category_preds[infalted_mask]) != 0:
                    infalted_category_preds[infalted_mask] = inds_to_remove[0]

        # Shuffle and downsample depth pixels
        coords = torch.stack((uu, vv), dim=1)  # pixel pairs vector
        coords = coords[torch.randperm(coords.size()[0])]
        coords = coords[:int(coords.size(dim=0)/downsampling_factor)]
        # ordering back coords (useful for memory management in raytracing)
        sorted_coords, indices = torch.sort(coords[:, 1], stable=True)
        coords = coords[indices]
        sorted_coords, indices = torch.sort(coords[:, 0], stable=True)
        coords = coords[indices]
        
        uu = coords[:, 0]
        vv = coords[:, 1]
        xx = (vv - cx) * depth[uu, vv] / fx
        yy = (uu - cy) * depth[uu, vv] / fy
        zz = depth[uu, vv] / depth_factor

        features = features_per_pixels[0, :, uu, vv]
        features_cpu = features.detach().cpu().numpy()
        color = color_img[uu, vv, :]
        color_cpu = color.detach().cpu().numpy()

        pointcloud = torch.cat((xx.unsqueeze(1), yy.unsqueeze(1), zz.unsqueeze(1)), 1)
        uu, vv = uu.cpu().detach().numpy(), vv.detach().cpu().numpy()

        if (inds_to_remove is not None) and inds_to_remove!=[]:
            return pointcloud, features_cpu.T, color_cpu, infalted_category_preds[uu, vv], pointcloud_out
        else:
             return pointcloud, features_cpu.T, color_cpu, category_preds, pointcloud_out


def from_depth_to_pc(depth, calib_matrix, min_depth = 0.2, max_depth = 6.0, depth_factor=1., downsample_factor=10):
        """
        Creates the 3D pointcloud, in camera frame, from the depth.
        Uses CPU

        :param depth: matrix of shape (W , H), depth image from the camera
        :param calib_matrix: matrix of shape (3, 3) containing the intrinsic parameters of the camera in matrix form
        :param min_depth: (float) filters out the points below this Z distance: must be positive
        :param max_depth: (float) filters out the points above this Z distance:  must be positive
        :param depth_factor: (float) scale factor for the depth image (it divides the depth z values)
        :param downsample_factor: (float) how much to reduce the number of points extracted from depth
        :return: array of shape (N, 3)
        """
        #fx, fy, cx, cy = intrinsics
        fx = calib_matrix[0, 0]
        fy = calib_matrix[1, 1]
        cx = calib_matrix[0, 2]
        cy = calib_matrix[1, 2]

        if min_depth < 0.0:
              min_depth = 0.2
        if max_depth < 0.0:
              max_depth = 6.0
                
        h, w = depth.shape
        points = np.zeros([h*w , 3])
        count = 0
        for u in range(0, h):
            for v in range(0, w):
                z = depth[u, v]
                if (z > min_depth and z < max_depth): # filter depth based on Z
                    z = z / depth_factor
                    x = ((v - cx) * z) / fx
                    y = ((u - cy) * z) / fy
                    points[count] = [x, y, z]
                    count += 1
        np.resize(points, count)

        #Downsample
        points=points[(np.random.randint(0, points.shape[0], np.round(count/downsample_factor, 0).astype(int)) )]
        return points


