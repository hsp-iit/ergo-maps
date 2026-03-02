# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import os
from pathlib import Path
import signal
import threading

import torchvision.transforms as transforms
import numpy as np
import torch
import gdown
import time

from visual_language_navigation.utils.lseg_utils import get_lseg_feat
from visual_language_navigation.lseg.modules.models.lseg_net import LSegEncNet
from visual_language_navigation.utils.camera_utils import project_depth_features_pc_torch
from visual_language_navigation.utils.conversion_utils import quaternion_matrix, xyzrgb_array_to_pointcloud2, numpy_to_ros2_image
from visual_language_navigation.utils.mapping_utils import raycasting, publish_static_transform
from visual_language_navigation.map.semantic_map_server import SemanticMapServer

#ROS2 stuff
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from ros2_vlmaps_interfaces.srv import EnableMapping, CameraRGB
import rclpy.time
from rclpy.executors import MultiThreadedExecutor
from ros2_vlmaps_interfaces.srv import IndexMap
from rclpy.callback_groups import ReentrantCallbackGroup

# For FCCLIP
import importlib
# Debug print
import matplotlib.pyplot as plt
import io
from PIL import Image as ImagePIL


class SemanticMapBuilder(Node):
    def __init__(
        self
    ):
        super().__init__('map_builder_node')
        ### ROS2 Param declaration
        self.declare_parameters(
            namespace='',
            parameters=[
                ('img_topic_name', "/camera/rgbd/img"),         # ergocub: /camera/rgbd/img
                ('depth_topic_name', "/camera/rgbd/depth"),     # ergocub: /camera/rgbd/depth
                ('amcl_pose_topic', "/amcl_pose"),
                ('use_camera_info_topic', True),
                ('camera_info_topic', "/camera/rgbd/camera_info"),   # ergocub: /camera/rgbd/camera_info
                ('cam_calib_mat', [386.0, 0.0, 321.0, 0.0, 386.0, 238.0, 0.0, 0.0, 1.0]),
                ('maximum_height', 2.3),
                ('cell_size', 0.02),        #0.05
                ('grid_size', 1500),        #1000
                ('map_save_dir', "/home/user1"),
                ('map_name', "semantic_map.h5df"),
                ('amcl_cov_threshold', 0.04),      #0.015
                ('voxmap_index_result', "/map_index_result"),
                ('map_pointcloud_topic', "/voxmap"),
                ('target_frame', "map"),
                ('semantic_map_frame', 'vlmap'),
                ('use_raycasting', True),
                ('use_all_depth_for_raycast', True),
                ('voxel_offset', 0),
                ('raycast_distance_threshold', 0.5),
                ('raycast_camera_batch_size', 2**11),
                ('raycast_map_batch_size', 2**10),
                ('max_camera_distance', 3.0),
                ('depth_downsampling', 10),     #10 for resolution 640x480 36 for res 1280x720
                ('seg_vis', False),
                ('enable_mapping', True),
                ('use_feature_fusion', False),
                ('lseg_use_cosine_sim', True),
                ('lseg_cossim_threshold', 0.9),
                ('camera_reference_frame', "realsense_compensated"),  # Reference frame of the camera, if empty will use the one from the ros message
                ('robot_base_frame','geometric_unicycle'),  # mobile_base_body_link for R1, geometric_unicycle for ergoCub
                ('classes_to_skip', ["person", "floor", "ceiling", "wall"]),
                ('labels', [
                            "screen"
                            ,"photo"
                            ,"laptop"
                            ,"wardrobe"
                            ,"table"
                            ,"closet"
                            ,"chair"
                            ,"shelf"
                            ,"door"
                            ,"wall"
                            ,"ceiling"
                            ,"floor"
                            ,"robot"
                            ]),
                ('seg_model_name', "fcclip"),   #fcclip
                ('debug', False),
                ('debug_category_list', [
                            "chair", "table", "floor", "door", "wall", "ceiling", "box", "television", "laptop", "shelving", "person", "trash bin" "first aid kit", "table",
                            "mouse", "hard hat", "printer", "microwave", "backpack", "bottle", "lunch bag", "banana", "remote", "smartphone", "cup", "yogurt", "drill", "tools", "glove",
                            ]),
                ('costmap_topic_name', "/global_costmap/costmap"),
                ('map_indexing_result_topic', "/map_2d_index_marker"),
                ('map_indexing_goal_topic', "/map_goal_indexing"),
                ('occupied_threshold', 40.0),
                ('costmap_goal_window', 5),
                ('costmap_floor_height', 0.2),
                ('costmap_dilation', 3),
                ('map_frame', "map"),
                ("dbscan.0.eps", 2.0),
                ("dbscan.0.min_samples", 10),
                ("dbscan.1.eps", 5.0),
                ("dbscan.1.min_samples", 80),
                ("dbscan.2.eps", 10.0),
                ("dbscan.2.min_samples", 200),
                ("dbscan.3.eps", 10.0),
                ("dbscan.3.min_samples", 300),
                ("dbscan.default.eps", 3.0),
                ("dbscan.default.min_samples", 10),
                ('batch_size_indexing', 2**14),
                ('fcclip_m2f_cos_sim_threshold', 0.22),
                ('fcclip_clip_cos_sim_threshold', 0.25)
                ])
        # Name of the rgb image topic
        img_topic = self.get_parameter('img_topic_name').value
        # Name of the depth image topic
        depth_topic = self.get_parameter('depth_topic_name').value
        # Boolean flag to wether read the calibration matrix from topic (if true) or from parameters
        use_camera_info_topic = self.get_parameter('use_camera_info_topic').value
        # Name of the camera_info topic
        camera_info_topic = self.get_parameter('camera_info_topic').value
        # Frame to which transform the pointcloud/map. By default is: map
        self.target_frame = self.get_parameter('target_frame').value
        # AMCL covariance threshold. At higher values the mapping is avoided.
        self.amcl_cov_threshold = self.get_parameter('amcl_cov_threshold').value
        if self.amcl_cov_threshold < 0:
            self.amcl_cov_threshold = 0.01
        # Name of the AMCL pose topic
        amcl_pose_topic = self.get_parameter('amcl_pose_topic').value
        # Maximum height of the map, in meters
        maximum_height = self.get_parameter('maximum_height').value
        # Cell/voxel size of the map, in meters
        cs = self.get_parameter('cell_size').value
        # N x N size of the voxel grid, N is the number of voxels
        gs = self.get_parameter('grid_size').value
        # Map save directory path name
        self.map_save_dir = self.get_parameter('map_save_dir').value
        self.map_save_path = self.map_save_dir + "/" + self.get_parameter('map_name').value # TODO check for file extension presence
        # Maximum distance for depth, points at higher range will be discarted
        self.max_depth = self.get_parameter('max_camera_distance').value
        # Downsampling factor for the camera pointcloud
        self.depth_downsampling = self.get_parameter('depth_downsampling').value
        os.makedirs(self.map_save_dir, exist_ok=True)
        # Boolean flag to whether use the raycasting and clear map voxels, or not
        self.use_raycast = self.get_parameter('use_raycasting').value
        # Flag to wheter use for raycasting also the points exceeding >= max depth param
        self.use_all_depth_for_raycast = self.get_parameter('use_all_depth_for_raycast').value
        # (float) Number of voxels before the final point of the pointcloud to stop raycasting.
        # An higher value means a 
        self.voxel_offset = self.get_parameter('voxel_offset').value
        # Distance (in voxels unit) from a ray to consider a voxel intercepted, and thus to be removed from the map.
        self.raycast_distance_threshold = self.get_parameter('raycast_distance_threshold').value
        # Batch size for the camera pointcloud (for raycasting computing memory limit)
        self.camera_batch_sz = self.get_parameter('raycast_camera_batch_size').value
        # Batch size for the map pointcloud (for raycasting computing memory limit)
        self.map_batch_sz = self.get_parameter('raycast_map_batch_size').value
        # How to call the voxel map frame
        self.semantic_map_frame_name = self.get_parameter('semantic_map_frame').value
        # Flag that if true accepts incoming img and depth messages for mapping
        self.enable_mapping = self.get_parameter('enable_mapping').value
        # Publishes (for debug) the segmentation image on a ros topic semantic_map_builder/seg_image
        self.seg_vis = self.get_parameter('seg_vis').value
        # List of strings of the classes to skip mapping. The points matching these categories will not be saved.
        self.classes_to_skip = self.get_parameter('classes_to_skip').value
        # List of strings of classes to map.
        self.labels = self.get_parameter('labels').value
        # Boolean flag. If true uses feature fusion for color and embedding values on repeated voxels
        self.use_feature_fusion = self.get_parameter('use_feature_fusion').value
        # Segmentation model used
        self.seg_model_name = self.get_parameter('seg_model_name').value
        # do debug print (slower)
        self.debug = self.get_parameter('debug').value
        # category list to display frame by frame
        self.debug_category_list = self.get_parameter('debug_category_list').value
        # Reference frame of the camera, if empty will use the one from the ros message
        self.camera_reference_frame = self.get_parameter('camera_reference_frame').value
        # Base frame of the robot, counts as the robot pose
        self.robot_base_frame = self.get_parameter("robot_base_frame").value

        ####################################### SEMANTIC MAP SERVER UNIQUE PARAMETERS
        # If true uses cosine similarity instead of argmax to see if a voxel is matching the text query, used during map indexing.
        self.lseg_use_cosine_sim = self.get_parameter('lseg_use_cosine_sim').value
        # Threshold of lseg cosine similarity to consider a voxel matching the category
        self.lseg_cossim_threshold = self.get_parameter("lseg_cossim_threshold").value
        # name of the global costmap topic
        self.costmap_topic_name = self.get_parameter("costmap_topic_name").value
        # Name of the topic where the results of the indexing are published as markers
        self.map_indexing_result_topic = self.get_parameter("map_indexing_result_topic").value
        # Topic name of the goal marker
        self.map_indexing_goal_topic = self.get_parameter("map_indexing_goal_topic").value
        # Thresold under which a cell in a costmap is considered free. It's used to compute the goal in the map.
        self.occupied_threshold = self.get_parameter("occupied_threshold").value
        # Number of voxels around the goal to extract the costmap from
        self.costmap_goal_window = self.get_parameter("costmap_goal_window").value
        # Height in meters after which we consider a voxel to be an obstacle
        self.costmap_floor_height = self.get_parameter("costmap_floor_height").value
        # Dilation kernel dimension to apply to the voxels projected to the costmap to be considered obstacles
        self.costmap_dilation = self.get_parameter("costmap_dilation").value
        # Map frame name in the ros2 navigation stack
        self.map_frame = self.get_parameter("map_frame").value
        # Clustering parameters for DBSCAN based on object sizes: [0, 1, 2, 3] 0: small -> 3:very large
        self.dbscan_0_eps = self.get_parameter("dbscan.0.eps").value
        self.dbscan_0_min_samples = self.get_parameter("dbscan.0.min_samples").value
        self.dbscan_1_eps = self.get_parameter("dbscan.1.eps").value
        self.dbscan_1_min_samples = self.get_parameter("dbscan.1.min_samples").value
        self.dbscan_2_eps = self.get_parameter("dbscan.2.eps").value
        self.dbscan_2_min_samples = self.get_parameter("dbscan.2.min_samples").value
        self.dbscan_3_eps = self.get_parameter("dbscan.3.eps").value
        self.dbscan_3_min_samples = self.get_parameter("dbscan.3.min_samples").value
        self.dbscan_default_eps = self.get_parameter("dbscan.default.eps").value
        self.dbscan_default_min_samples = self.get_parameter("dbscan.default.min_samples").value
        # Batch size of number of map voxels to index/query in batches
        self.batch_size_indexing = self.get_parameter("batch_size_indexing").value
        # Cosine similarity threshold for mask2former
        self.fcclip_m2f_cos_sim_threshold = self.get_parameter("fcclip_m2f_cos_sim_threshold").value
        # Cosine similarity threshold for the CLIP feature matching component of fcclip
        self.fcclip_clip_cos_sim_threshold = self.get_parameter("fcclip_clip_cos_sim_threshold").value
        # For video only: show an asked category while publishing the map
        self.show_object_on_map_srv = self.create_service(IndexMap, 'semantic_map_server/show_object_on_map', self.show_object_callback)
        self.callback_group = ReentrantCallbackGroup() 
        self.index_client = self.create_client(IndexMap, 'semantic_map_server/index_map', callback_group=self.callback_group)
        self.show_indexing = False

        self.get_logger().info(f'Using parameters: {img_topic=}  {depth_topic=}  {use_camera_info_topic=}  {camera_info_topic=} \n'
                               f'{self.target_frame=}  {self.amcl_cov_threshold=}  {amcl_pose_topic=}  {maximum_height=}  {self.max_depth=}  {self.depth_downsampling=}\n'
                               f'{cs=}  {gs=}  {self.map_save_path=}  {self.use_raycast=}  {self.camera_batch_sz=}  {self.map_batch_sz=} \n'
                               f'{self.voxel_offset=}  {self.raycast_distance_threshold=}  {self.semantic_map_frame_name=} \n'
                               f'{self.enable_mapping=}  {self.seg_vis=}  {self.classes_to_skip=}  {self.labels=} {self.robot_base_frame=} \n'
                               f'{self.use_feature_fusion=}  {self.lseg_use_cosine_sim=} {self.seg_model_name=} {self.debug=} {self.debug_category_list=}')

        # Enable camera ingo subscriber or load params
        if use_camera_info_topic == True:
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                camera_info_topic,
                self.camera_info_callback,
                10
            )
            self.camera_info_available = False
        else:
            self.calib_mat = np.array(self.get_parameter('cam_calib_mat').value).reshape((3, 3))
            self.camera_info_available = True
        
        ### tf2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.static_tf_published = False
        ### ROS2 subscribers
        self.img_sub = message_filters.Subscriber(self, Image, img_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.tss = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], 1, slop=0.3)        
        self.tss.registerCallback(self.sensors_callback)
        self.amcl_callback_group = ReentrantCallbackGroup() 
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            amcl_pose_topic,
            self.amcl_callback,
            10,
            callback_group=self.amcl_callback_group
        )
        ### ROS2 publishers
        if self.seg_vis:
            self.seg_img_pub = self.create_publisher(Image, "semantic_map_builder/seg_image", 1)
            self.masks_img_pub = self.create_publisher(Image, "semantic_map_builder/masks_of_image", 1)

        ### ROS2 services
        self.enable_mapping_srv = self.create_service(EnableMapping, 'semantic_map_builder/enable_mapping', self.enable_mapping_callback)
        self.camera_echo_srv = self.create_service(CameraRGB, "semantic_map_builder/robot_camera_rgb", self.camera_callback)

        ### Segmentation of Classes we want to avoid mapping
        self.get_preds = False  #flag that specifies if we need the predictions of the masks or image predictions
        self.inds_to_remove = []
        if self.classes_to_skip is not None and len(self.classes_to_skip) > 0:
            self.get_preds = True
            # Add these classes to the lables
            if len(self.labels) > 0:
                classes_to_add = []
                for i in range(len(self.classes_to_skip)):
                    found = False
                    for j in range(len(self.labels)):
                        if self.classes_to_skip[i] == self.labels[j]:
                            found = True
                            break
                    if not found:
                        classes_to_add.append(self.classes_to_skip[i])
                if len(classes_to_add) > 0:
                    self.labels.extend(classes_to_add)
            else:
                self.labels = self.classes_to_skip
        
            # Find the indices of the classes to remove: we will ignore the matching pixels during mapping
            for k in range(len(self.classes_to_skip)):
                for j in range(len(self.labels)):
                    if self.classes_to_skip[k] == self.labels[j]:
                        self.inds_to_remove.append(j)
                        break
        
        ### init language segmentation model
        if self.seg_model_name == "lseg":
            self.seg_model, self.lseg_transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std = self._init_lseg()
        elif self.seg_model_name == "fcclip":
            # We import dynamically fcclip here, so we don't need fclip always installed
            self.fcclip_init_mod = importlib.import_module("visual_language_navigation.utils.fcclip_utils")
            self.get_fcclip_feat = self.fcclip_init_mod.get_fcclip_feat
            self.init_fcclip = self.fcclip_init_mod._init_fcclip
            self.seg_model, self.lseg_transform = self.init_fcclip()
            self.seg_model.preencode_text(self.labels)
            self.clip_feat_dim = 768 + 768  #m2f and clip
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.get_logger().error("Segmentation model not found")
            exit(1)

        ### init the map
        self.map_wrapper = SemanticMapServer("voxmap_srv_node")
        # Pass shared parameters to map wrapper
        params_list = [self.get_parameter('map_name'),
                       self.get_parameter('map_save_dir'),
                       self.get_parameter('semantic_map_frame'),
                       self.get_parameter('voxmap_index_result'),
                       self.get_parameter('map_pointcloud_topic'),
                       self.get_parameter('cell_size'),
                       self.get_parameter('grid_size'),
                       self.get_parameter('lseg_use_cosine_sim'),
                       self.get_parameter('labels'),
                       self.get_parameter('seg_model_name'),
                       self.get_parameter('robot_base_frame'),
                       self.get_parameter("lseg_cossim_threshold"),
                       self.get_parameter("costmap_topic_name"),
                       self.get_parameter("map_indexing_result_topic"),
                       self.get_parameter("map_indexing_goal_topic"),
                       self.get_parameter("occupied_threshold"),
                       self.get_parameter("costmap_goal_window"),
                       self.get_parameter("costmap_floor_height"),
                       self.get_parameter("costmap_dilation"),
                       self.get_parameter("map_frame"),
                       self.get_parameter("dbscan.0.eps"),
                       self.get_parameter("dbscan.0.min_samples"),
                       self.get_parameter("dbscan.1.eps"),
                       self.get_parameter("dbscan.1.min_samples"),
                       self.get_parameter("dbscan.2.eps"),
                       self.get_parameter("dbscan.2.min_samples"),
                       self.get_parameter("dbscan.3.eps"),
                       self.get_parameter("dbscan.3.min_samples"),
                       self.get_parameter("dbscan.default.eps"),
                       self.get_parameter("dbscan.default.min_samples"),
                       self.get_parameter("batch_size_indexing"),
                       self.get_parameter("fcclip_m2f_cos_sim_threshold"),
                       self.get_parameter("fcclip_clip_cos_sim_threshold"),
                       ]
        self.map_wrapper.set_parameters(params_list)
        self.map_wrapper.initialize(self.seg_model)   # init node with the parameters, has to be done after set_parameters
        # Initialize voxel map (load or create a new one)
        self.loaded_map = self.map_wrapper.voxel_map.init_map(maximum_height, self.map_save_path, self.clip_feat_dim)

        self.is_first_iter = True           # Flag if is the first iteration of the mapping callback (used for deciding when to raycast)
        self.missed_frame_i = 0
        self.amcl_cov = np.full([36], 0.2)  # init with a high covariance value for each element

        self.img_lock = threading.Lock()
        self.img_msg = None

        # Handle Exit signal. Save map at exit.
        self.save_lock = threading.Lock()
        signal.signal(signal.SIGINT, self.exit_handle)

        # Set PYTORCH cuda allocation to not be aggressive
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'garbage_collection_threshold:0.6,max_split_size_mb:32'
    
    def enable_mapping_callback(self, request : EnableMapping.Request, response : EnableMapping.Response):
        """
        Service callback that stores the boolean flag
        """
        try:
            self.enable_mapping = request.enable_flag
            response.is_ok = True
        except Exception as ex:
            response.is_ok = False
            response.error_msg = f"[enable_mapping_callback] An exception occurred {ex=}"
        return response
    
    def sensors_callback(self, img_msg : Image, depth_msg : Image):
        """
        build the 3D map centered at the first base frame
        """
        self.missed_frame_i += 1
        # TODO remove in favour of creating the service directly on the camera device driver
        with self.img_lock:
            self.img_msg = img_msg  # Save for ros2 service when asked about the robot sight by LLM
        if not self.enable_mapping:
            print("Mapping not enabled: skipping callback")
            return
        if not self.camera_info_available:
            print("Waiting for camera info topic to be available: \nIf you don't want this feature, disable it in the config file")
            return

        loop_timer = time.time()
        #### first do a TF check between the camera and map frame
        try:
            if self.camera_reference_frame != "" and self.camera_reference_frame is not None:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.camera_reference_frame,
                    depth_msg.header.stamp
                    )
            else:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    depth_msg.header.frame_id,
                    depth_msg.header.stamp
                    )

        except TransformException as ex:
                if self.camera_reference_frame != "" or self.camera_reference_frame is not None:
                    self.get_logger().info(
                            f'Could not transform {self.camera_reference_frame} to {self.target_frame}: {ex}')
                else:
                    self.get_logger().info(
                            f'Could not transform {depth_msg.header.frame_id} to {self.target_frame}: {ex}')
                return
        #### Check amcl covariance
        if not (abs(self.amcl_cov.max()) < self.amcl_cov_threshold) and (abs(self.amcl_cov.min()) < self.amcl_cov_threshold):
            self.get_logger().info(f'Covariance too big: skipping callback until amcl converges')
            return

        #### Convert tf2 transform to np array components
        transform_pose_np = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        transform_quat_np = np.array([transform.transform.rotation.x, transform.transform.rotation.y,
                                        transform.transform.rotation.z, transform.transform.rotation.w])
        # Let's get an SE(4) matrix form
        transform_np = quaternion_matrix(transform_quat_np)
        transform_np[0:3, -1] = transform_pose_np
        transform_torch = torch.tensor(transform_np, device=self.device, dtype=torch.float32)
        # Convert ROS Image message to NumPy array (raw byte data) and then to Tensor
        rgb = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        rgb_torch = torch.tensor(rgb, device=self.device)
        depth = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
        depth_torch = torch.tensor(depth, device=self.device)

        #### Segment image and extract features
        if self.seg_model_name == "fcclip":
            mask_feats, pix_indices, category_preds = self.get_fcclip_feat(self.seg_model, rgb, self.labels, self.lseg_transform, self.device, get_preds=self.get_preds)
            
            if self.debug or self.seg_vis:
                debug_mask_feats, debug_pix_indices, debug_category_preds = self.get_fcclip_feat(self.seg_model, rgb, self.debug_category_list, self.lseg_transform, self.device, get_preds=self.get_preds)

                if self.frame_i % 1 == 0:
                    pix_idx = torch.where(debug_pix_indices == -1, debug_mask_feats.shape[0], debug_pix_indices)
                    colors = plt.cm.nipy_spectral(np.linspace(0, 1, debug_mask_feats.shape[0] + 1))
                    label_index_to_color_dict = {i: torch.tensor(colors[i]) for i in range(debug_mask_feats.shape[0] + 1)}

                    # Initialize open_seg_mask
                    open_seg_mask = torch.zeros(pix_idx.shape[0], pix_idx.shape[1], 4)
                    # Update open_seg_mask using tensor indexing
                    open_seg_mask = torch.stack([label_index_to_color_dict[idx.item()] for idx in pix_idx.flatten()]).view(open_seg_mask.shape)
                    fig, ax = plt.subplots()
                    plt.imshow(rgb)
                    plt.imshow(open_seg_mask, alpha=.5)
                    detected_categories = torch.unique(pix_idx)
                    found_categories_legend = ax.legend(handles=[plt.Rectangle((0,0),1,1, color=np.array(label_index_to_color_dict[int(i)]),                                                       
                                                            label=int(i)) for i in range(debug_mask_feats.shape[0] + 1) if i in detected_categories],
                                                            title = "Detected Masks",   
                                                            loc='upper left', bbox_to_anchor=(1, 1),
                                                            )

                    # Manually add the first legend back to the plot
                    fig.add_artist(found_categories_legend)
                    format = "png"
                    plt.title("FCCLIP")
                    plt.axis("off")
                    plt.tight_layout()
                    if self.seg_vis:
                        buf = io.BytesIO()
                        plt.savefig(buf, format=format, bbox_inches='tight')
                        buf.seek(0)
                        # Open the PNG image from the buffer and convert it to a NumPy array
                        seg_img = np.array(ImagePIL.open(buf))
                        # Close the buffer
                        buf.close()
                        img_msg = numpy_to_ros2_image(seg_img, img_msg.header.stamp, img_msg.header.frame_id)
                        self.masks_img_pub.publish(img_msg)

                    if self.debug:
                        os.makedirs(os.path.join(os.getcwd(), 'fcclip_frames', 'masks'), exist_ok=True)
                        plt.savefig(f'fcclip_frames/masks/fcclip{self.missed_frame_i}.{format}', format=format, bbox_inches='tight')

                    #################################################################################################

                    fig, ax = plt.subplots()
                    labels = [", ".join([w]) for w in self.debug_category_list]
                    labels.append("zero embedding")
                    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(labels)))
                    label_index_to_color_dict = {i: torch.tensor(colors[i]) for i in range(len(labels))}
                    cat_preds = torch.where(debug_category_preds == -1, len(self.debug_category_list), debug_category_preds)
                    open_seg_mask = torch.zeros(cat_preds.shape[0], cat_preds.shape[1], 4)

                    # Update open_seg_mask using tensor indexing
                    open_seg_mask = torch.stack([label_index_to_color_dict[idx.item()] for idx in cat_preds.flatten()]).view(open_seg_mask.shape)

                    plt.imshow(rgb)
                    plt.imshow(open_seg_mask, alpha=.5)
                    detected_categories = torch.unique(cat_preds)
                    found_categories_legend = ax.legend(handles=[plt.Rectangle((0,0),1,1, color=np.array(label_index_to_color_dict[int(i)]),                                                       
                                                            label=labels[int(i)]) for i in range(len(labels)) if i in detected_categories],
                                                            title = "Detected categories",   
                                                            loc='upper left', bbox_to_anchor=(1, 1),
                                                            )

                    missing_categories_legend = plt.legend(handles=[plt.Rectangle((0,0),1,1, color=np.array(label_index_to_color_dict[int(i)]),
                                                            label=labels[int(i)]) for i in range(len(labels)) if i not in detected_categories],
                                                            title= "Missing categories",
                                                            loc='upper right', bbox_to_anchor=(0, 1),
                                                            )

                    # Manually add the first legend back to the plot
                    fig.add_artist(found_categories_legend)
                    format = "png"
                    plt.title("FC-CLIP")
                    plt.axis("off")
                    plt.tight_layout()

                    if self.seg_vis:
                        buf = io.BytesIO()
                        plt.savefig(buf, format=format, bbox_inches='tight')
                        buf.seek(0)

                    if self.debug:
                        os.makedirs(os.path.join(os.getcwd(), 'fcclip_frames', 'segm'), exist_ok=True)
                        plt.savefig(f'fcclip_frames/segm/fcclip{self.missed_frame_i}.{format}', format=format, bbox_inches='tight')

                    if self.seg_vis:
                        # Open the PNG image from the buffer and convert it to a NumPy array
                        seg_img = np.array(ImagePIL.open(buf))
                        # Close the buffer
                        buf.close()

                        img_msg = numpy_to_ros2_image(seg_img, img_msg.header.stamp, img_msg.header.frame_id)
                        self.seg_img_pub.publish(img_msg)

            category_preds_cpu = category_preds.squeeze().detach().cpu().numpy().astype(int)
            pix_feats = pix_indices.permute(2, 0, 1).unsqueeze(0) # To make it compatible with LSEG
            mask_feats = mask_feats.detach().cpu().numpy()
            
        # Pixel aligned LSeg (CLIP) features
        elif self.seg_model_name == "lseg":
            # Flag to show realtime the segmentation of the camera with the given lables
            if self.seg_vis:
                pix_feats, category_preds_cpu, seg_img = get_lseg_feat(
                    self.seg_model,
                    rgb, self.labels,
                    self.lseg_transform,
                    self.device,
                    self.crop_size,
                    self.base_size,
                    self.norm_mean,
                    self.norm_std,
                    self.classes_to_skip,
                    vis=self.seg_vis
                    )
                img_msg = numpy_to_ros2_image(seg_img, img_msg.header.stamp, img_msg.header.frame_id)
                self.seg_img_pub.publish(img_msg)
            else:
                pix_feats, category_preds_cpu = get_lseg_feat(
                    self.seg_model, 
                    rgb, self.labels,
                    self.lseg_transform,
                    self.device,
                    self.crop_size,
                    self.base_size,
                    self.norm_mean,
                    self.norm_std,
                    self.classes_to_skip,
                    vis=self.seg_vis
                )

        #### Formatted PC with aligned features per pixel
        camera_pointcloud_xyz, features_per_point, color_per_point, category_preds, outer_points = project_depth_features_pc_torch(depth_torch, 
                                                                                                                                   pix_feats, 
                                                                                                                                   rgb_torch, 
                                                                                                                                   self.calib_mat, 
                                                                                                                                   self.inds_to_remove, 
                                                                                                                                   category_preds_cpu, 
                                                                                                                                   max_depth = self.max_depth, 
                                                                                                                                   downsampling_factor = self.depth_downsampling,
                                                                                                                                   kernel_dilation_size = 20)
        if (category_preds is None) or (not self.get_preds):
            category_preds = np.full_like(rgb_torch, -1)

        #### Transform PC into map frame
        pc_global = torch.hstack([camera_pointcloud_xyz, torch.ones((camera_pointcloud_xyz.shape[0], 1), dtype=torch.float32, device='cuda')]) @ transform_torch.T
        pc_global = pc_global[:, 0:3]
        pc_global_cpu = pc_global.detach().cpu().numpy()
        # Transform also the points removed (used for a more aggressive raycasting)
        if self.use_all_depth_for_raycast:
            outer_points = torch.hstack([outer_points, torch.ones((outer_points.shape[0], 1), dtype=torch.float32, device='cuda')]) @ transform_torch.T
            outer_points = outer_points[:, 0:3]
        #### Raycast
        try:
            cam_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            if (not (self.is_first_iter) or (self.loaded_map == True)) and self.use_raycast:
                if self.use_all_depth_for_raycast & (outer_points.shape[0] != 0):
                    full_pc = torch.cat([pc_global, outer_points])
                else:
                    full_pc = pc_global
                voxels_to_clear = raycasting(cam_pose, 
                                             full_pc, 
                                             self.map_wrapper.voxel_map.grid_size, 
                                             self.map_wrapper.voxel_map.cell_size, 
                                             self.map_wrapper.voxel_map.grid_pos, 
                                             self.map_wrapper.voxel_map.voxels_flags, 
                                             self.raycast_distance_threshold, 
                                             self.voxel_offset, 
                                             self.camera_batch_sz, 
                                             self.map_batch_sz, 
                                             self.device)
                
                self.map_wrapper.voxel_map.remove_map_voxels(voxels_to_clear)
        except Exception as ex:
            self.get_logger().error(f"Caught unexpected exception while raycasting: {ex}\n Skipping this mapping iteration")
            return
        
        #### Map update
        if self.seg_model_name == "fcclip":
            features_per_point = (mask_feats, features_per_point)
        self.map_wrapper.voxel_map.update_map(pc_global_cpu, features_per_point, color_per_point, category_preds, self.inds_to_remove, self.use_feature_fusion, cam_pose)
        self.is_first_iter = False
        
        self.get_logger().info(f"CALLBACK TIME: {time.time() - loop_timer}")

        #### Map pointcloud publishing
        mask = (self.map_wrapper.voxel_map.grid_pos > 0).all(axis=1)
        occupied_mask = (self.map_wrapper.voxel_map.voxels_flags == 1)
        final_mask = mask & occupied_mask
        color = self.map_wrapper.voxel_map.grid_rgb[final_mask]
        
        points = self.map_wrapper.voxel_map.grid_pos[final_mask] * self.map_wrapper.voxel_map.cell_size  #scale it to meters
        msg = xyzrgb_array_to_pointcloud2(points, color, stamp=self.get_clock().now().to_msg(), frame_id=self.semantic_map_frame_name)
        msg.header.stamp = depth_msg.header.stamp
        if not self.static_tf_published:
            if publish_static_transform(depth_msg.header.stamp, self.target_frame, self.semantic_map_frame_name, self.map_wrapper.voxel_map.grid_size, self.map_wrapper.voxel_map.cell_size, self.tf_static_broadcaster):
                self.static_tf_published = True
        self.get_logger().info(f"MAX ID: {self.map_wrapper.voxel_map.max_id}")
        self.map_wrapper.map_pointcloud_pub.publish(msg)
        if self.show_indexing:
            request = IndexMap.Request()
            request.indexing_string = self.category_to_show
            self.index_client.call_async(request=request)
        return
    
    def show_object_callback(self, request : IndexMap.Request, response : IndexMap.Response):
        if request.indexing_string == "":
            self.show_indexing = False
        else:
            self.show_indexing = True
            self.category_to_show = request.indexing_string
        response.is_ok = True
        response.error_msg = ""
        return response

    # Simply store the covariance values, will be analyzed in the sensors callback
    def amcl_callback(self, msg : PoseWithCovarianceStamped):
        """
        Stores the covariance values of amcl pose
        """
        self.amcl_cov = msg.pose.covariance

    # Just do it once
    def camera_info_callback(self, msg : CameraInfo):
        """
        Saves the calib matrix of the camera intrinsic parameters
        """
        if not self.camera_info_available:
            self.calib_mat = np.array(msg.k, dtype=np.float32).reshape((3, 3))
            self.camera_info_available = True

    def _init_lseg(self, crop_size = 480, base_size = 640):
        #crop_size = 480  # 480
        #base_size = 640  # 520
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        lseg_model = LSegEncNet("", arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size)
        model_state_dict = lseg_model.state_dict()
        checkpoint_dir = Path(__file__).resolve().parents[1] / "lseg" / "checkpoints"
        checkpoint_path = checkpoint_dir / "demo_e200.ckpt"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"checkpoint path is : {checkpoint_path}")

        if not checkpoint_path.exists():
            print("Downloading LSeg checkpoint...")
            # the checkpoint is from official LSeg github repo
            # https://github.com/isl-org/lang-seg
            checkpoint_url = "https://drive.google.com/u/0/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
            gdown.download(checkpoint_url, output=str(checkpoint_path))

        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        pretrained_state_dict = {k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()}
        model_state_dict.update(pretrained_state_dict)
        lseg_model.load_state_dict(pretrained_state_dict)

        lseg_model.eval()
        lseg_model = lseg_model.to(self.device)

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        lseg_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.clip_feat_dim = lseg_model.out_c
        return lseg_model, lseg_transform, crop_size, base_size, norm_mean, norm_std
    
    # Callback that provides the image to the LLM Agent
    def camera_callback(self, request : CameraRGB.Request, response : CameraRGB.Response):
        self.get_logger().info("Got a request to provide the camera image")
        with self.img_lock:
            if self.img_msg != None:
                response.rgb = self.img_msg
                response.error_msg = ''
                response.is_ok = True
            else:
                response.error_msg = 'Image not available'
                response.is_ok = False
                response.rgb = None
        return response
    
    # Safely save the map before exiting
    def exit_handle(self, frame, _):
        print("Detected Ctrl+C: saving map and closing...")
        with self.save_lock:
            self.map_wrapper.voxel_map.save_map(self.map_save_path)
            # Cleanup
            print("Shutting down ros node")
            self.destroy_node()
            rclpy.shutdown()
            exit(0)


def main():
    rclpy.init()
    node = SemanticMapBuilder()
    print(f"Created {node.get_name()}")
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    exe.add_node(node.map_wrapper)
    print(f"Spinning Node {node.get_name()}")
    exe.spin()

if __name__ == "__main__":
    main()
