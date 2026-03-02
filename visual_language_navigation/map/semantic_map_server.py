# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import sys
import rclpy
from rclpy.node import Node
from ros2_vlmaps_interfaces.srv import IndexMap, LoadMap, ShowMap, LlmQuery, PublishGoal, EvaluateMap
from ros2_vlmaps_interfaces.msg import VoxelObject, ObjectQueryResult
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

from visual_language_navigation.map.voxel_feat_map import VoxelFeatMap

from sklearn.cluster import DBSCAN
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visual_language_navigation.utils.clip_utils import init_clip
from visual_language_navigation.utils.conversion_utils import quaternion_from_euler, xyzrgb_array_to_pointcloud2, convert_from_costmap_cell, convert_in_costmap_cell
from visual_language_navigation.utils.mapping_utils import from_grid_coords, find_closest_free_cell, publish_static_transform, to_grid_coords
from visual_language_navigation.utils.cluster_utils import pub_clusters_verticies, compute_cluster_centers
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration

import copy
import cv2
import importlib
import time

class SemanticMapServer(Node):
    def __init__(self, 
                 node_name: str
                 ) -> None:
        super().__init__(node_name)
        ### Param declaration
        self.declare_parameters(
            namespace='',
            parameters=[
                ('costmap_topic_name', "/global_costmap/costmap"),
                ('map_indexing_result_topic', "/map_2d_index_marker"),
                ('map_indexing_goal_topic', "/map_goal_indexing"),
                ('map_pointcloud_topic', "/voxmap"),
                ('voxmap_index_result', "/map_index_result"),
                ('cell_size', 0.02),
                ('grid_size', 1500),
                ('map_save_dir', "/home/user1"),
                ('map_name', "semantic_map.h5df"),
                ('semantic_map_frame', 'vlmap'),
                ('occupied_threshold', 40.0),
                ('costmap_goal_window', 5),
                ('costmap_floor_height', 0.2),
                ('costmap_dilation', 3),
                ('lseg_cossim_threshold', 0.9),
                ('lseg_use_cosine_sim', True),
                ('robot_base_frame','geometric_unicycle'),  # mobile_base_body_link for R1, geometric_unicycle for ergoCub
                ('seg_model_name', "fcclip"),       # fcclip or lseg
                ('labels', ["other"
                            ,"screen"
                            ,"table"
                            ,"closet"
                            ,"chair"
                            ,"shelf"
                            ,"door"
                            ,"wall"
                            ,"ceiling"
                            ,"floor"
                            ,"human"]),
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
        
        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        # Rotation matrix for aligning the semantic map with the nav2 map
        self.rot_mat_z = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, 1]])
        
        # Vars init
        self.use_colors = False
        self.tf_published = False
        self.costmap_ready = False
        # Array of markers of the indexing result
        self.markers_array = MarkerArray()
        self.markers_array.markers = []
        self.initialized = False
        print(f"{node_name} started successfully")

    
    def initialize(self, seg_model):
        """
        Loads the parameters value from the server and initializes the services, subscribers and publishers of the node
        Initializes/loads the voxel map.
        """
        if not self.initialized:
            # Used only by FCCLIP
            self.seg_model = seg_model
            # Base frame of the robot, i.e. the reference frame
            self.robot_base_frame = self.get_parameter('robot_base_frame').value
            # name of the global costmap topic
            costmap_topic_name = self.get_parameter('costmap_topic_name').value
            # Name of the topic where the results of the indexing are published as markers
            map_indexing_result_topic = self.get_parameter('map_indexing_result_topic').value
            # Topic name of the goal marker
            map_indexing_goal_topic = self.get_parameter('map_indexing_goal_topic').value
            # Topic name of the voxel map as pointcloud
            map_pointcloud_topic = self.get_parameter('map_pointcloud_topic').value
            # Topic name where the results of the indexing are published as pointcloud
            index_pointcloud_topic = self.get_parameter('voxmap_index_result').value
            # Cell/voxel size of the map, in meters
            self.cell_size = self.get_parameter('cell_size').value
            # N x N size of the voxel grid, N is the number of voxels
            self.grid_size = self.get_parameter('grid_size').value
            # Map save directory path name
            self.map_save_dir = self.get_parameter('map_save_dir').value
            self.map_save_path = self.map_save_dir + "/" + self.get_parameter('map_name').value
            # Name of the voxel map frame
            self.semantic_map_frame_name = self.get_parameter('semantic_map_frame').value
            # Thresold under which a cell in a costmap is considered free.
            # It's used to compute the goal in the map.
            self.occupied_threshold = self.get_parameter('occupied_threshold').value
            # Number of voxels around the goal to extract the costmap from
            self.costmap_goal_window = self.get_parameter('costmap_goal_window').value
            # Height in meters after which we consider a voxel to be an obstacle
            self.costmap_floor_height = self.get_parameter('costmap_floor_height').value
            # Dilation kernel dimension to apply to the voxels projected to the costmap to be considered obstacles
            self.costmap_dilation = self.get_parameter('costmap_dilation').value
            # Flag to enable lseg cosine similarity instead of argmax comparison with 'other'
            self.lseg_use_cosine_sim = self.get_parameter('lseg_use_cosine_sim').value
            # Threshold of lseg cosine similarity to consider a voxel matching the category
            self.lseg_cossim_threshold = self.get_parameter('lseg_cossim_threshold').value
            # Cosine similarity threshold for mask2former
            self.fcclip_m2f_cos_sim_threshold = self.get_parameter('fcclip_m2f_cos_sim_threshold').value
            # Cosine similarity threshold for the CLIP feature matching component of fcclip
            self.fcclip_clip_cos_sim_threshold = self.get_parameter('fcclip_clip_cos_sim_threshold').value
            # Batch size of number of map voxels to index/query in batches
            self.batch_size = self.get_parameter('batch_size_indexing').value
            # Type of segmentation model used
            self.seg_model_name = self.get_parameter('seg_model_name').value
            # Map frame name in the ros2 navigation stack
            self.map_frame = self.get_parameter('map_frame').value
            # DBSCAN setup
            # Build the dict from ROS params
            self.clustering_params = {
                size: {
                    "eps": self.get_parameter(f"dbscan.{size}.eps").value,
                    "min_samples": self.get_parameter(f"dbscan.{size}.min_samples").value
                }
                for size in ["0", "1", "2", "3", "default"]
            }

            # wether or not to save volatile memory during the construction of the map
            if self.seg_model_name == "fcclip":
                self.save_memory_at_runtime = True
            else:
                self.save_memory_at_runtime = False

            self.get_logger().info(f'Using Parameters: {costmap_topic_name=}  {map_indexing_result_topic=}  {map_indexing_goal_topic} \n'
                                   f'{map_pointcloud_topic=}  {index_pointcloud_topic=}  {self.map_save_path=}  {self.semantic_map_frame_name=} {self.grid_size=} {self.cell_size=}\n'
                                   f'{self.occupied_threshold=} {self.lseg_cossim_threshold=} {self.costmap_dilation=} {self.costmap_floor_height=} {self.costmap_goal_window=} \n'
                                   f'{self.seg_model_name=} {self.lseg_use_cosine_sim=} {self.save_memory_at_runtime=}')
            # Publishers Init
            self.map_indexing_marker_pub = self.create_publisher(MarkerArray, map_indexing_result_topic, 10)
            self.goal_marker_pub = self.create_publisher(Marker, map_indexing_goal_topic, 10)
            self.bbox_marker_pub = self.create_publisher(Marker, "/objects_bboxes", 10)
            self.map_pointcloud_pub = self.create_publisher(PointCloud2, map_pointcloud_topic, 10)
            self.index_pointcloud_pub = self.create_publisher(PointCloud2, index_pointcloud_topic, 10)
            self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
            # Sub init
            self.costmap_sub = self.create_subscription(OccupancyGrid, costmap_topic_name, self.costmap_callback, 10)
            # Service init
            self.index_map_srv = self.create_service(IndexMap, 'semantic_map_server/index_map', self.index_map_callback) 
            self.show_rgb_map_srv = self.create_service(ShowMap, 'semantic_map_server/show_semantic_map', self.show_rgb_map_callback) 
            self.load_map_srv = self.create_service(LoadMap, 'semantic_map_server/load_semantic_map', self.load_map_callback) 
            self.query_info_srv = self.create_service(LlmQuery, "semantic_map_server/llm_query", self.query_info_callback)
            self.goal_pub_srv = self.create_service(PublishGoal, "semantic_map_server/pub_goal", self.pub_goal_callback)
            self.eval_srv = self.create_service(EvaluateMap, "semantic_map_server/eval_map", self.eval_callback)
            
            # CLIP for LSEG
            if self.seg_model_name == "lseg":
                if hasattr(self, "clip_model"):
                    print("clip model is already initialized")
                else:
                    self.lseg_clip_model, self.lseg_clip_feat_dim = init_clip()
            # Map Loading
            self.voxel_map = VoxelFeatMap(self.seg_model, 
                                          cell_size=self.cell_size, 
                                          grid_size=self.grid_size, 
                                          save_memory_at_runtime=self.save_memory_at_runtime, 
                                          seg_model_name=self.seg_model_name)
            try:
                if self.voxel_map.load_map(self.map_save_path):
                    print(f"Map loaded successfully: {self.map_save_path}")
                else:
                    self.voxel_map.map_present = False
            except Exception as ex:
                print(f"[MapPublisher:__init__] An exception occurred: {ex=}")
            
            self.initialized = True
            self.get_logger().info(f"[{self.get_name()}] node initialized properly.")
        else:
            self.get_logger().warn(f"[{self.get_name()}] node already initialized, skipping.")

    
    def query_info_callback(self, request : LlmQuery.Request, response : LlmQuery.Response):
        if not self.initialized:
            self.get_logger().error(f"[query_info_callback] Node not initialized, aborting.")
            response.is_ok = False
            response.error_msg = f"Node not initialized. Call the method intialize() first."
            return response
        if not self.tf_published:
                publish_static_transform(stamp=self.get_clock().now().to_msg(),
                                         frame_id = self.map_frame,
                                         child_frame_id = self.semantic_map_frame_name,
                                         grid_size = self.voxel_map.grid_size,
                                         cell_size = self.voxel_map.cell_size,
                                         tf_static_broadcaster = self.tf_static_broadcaster)
        # Robot pose
        try:
            transform = self.tf_buffer.lookup_transform(
                    self.map_frame,              
                    self.robot_base_frame,  
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0)
                    )
            robot_pose = Pose()
            robot_pose.position.x = transform.transform.translation.x
            robot_pose.position.y = transform.transform.translation.y
            robot_pose.position.z = transform.transform.translation.z
            robot_pose.orientation = transform.transform.rotation
        except Exception as ex:
            self.get_logger().error(f"[query_info_callback] Robot pose not available, aborting. {ex=}")
            response.is_ok = False
            response.error_msg = f"[query_info_callback] Unable to get robot pose."
            return response
        object_substrings = request.object_string
        sizes_substrings = request.object_size.split(" ")
        start = time.time()
        # If we have less sizes specifications, we put the missing info to be medium sized objects
        if len(sizes_substrings) < len(object_substrings):
            for i in range (len(object_substrings) - len(sizes_substrings)):
                sizes_substrings.append("1")

        marker_iter_counter = 0
        for string, size in zip(object_substrings, sizes_substrings):
            if self.seg_model_name == "fcclip":
                mask = self.voxel_map.index_map_fcclip(language_desc=string,
                                                       fcclip_m2f_cos_sim_threshold=self.fcclip_m2f_cos_sim_threshold,
                                                       fcclip_clip_cos_sim_threshold=self.fcclip_clip_cos_sim_threshold,
                                                       batch_size=self.batch_size
                                                )
            elif self.seg_model_name == "lseg":
                mask = self.voxel_map.index_map_lseg(language_desc=string, 
                                            clip_model=self.lseg_clip_model, 
                                            lseg_clip_feat_dim=self.lseg_clip_feat_dim, 
                                            lseg_use_cosine_sim=self.lseg_use_cosine_sim, 
                                            lseg_clip_threshold=self.lseg_cossim_threshold,
                                            batch_size=self.batch_size
                                            )
            else:
                self.get_logger().error(f"[index_map_callback] Unsupported seg model: {self.seg_model_name=}")
                response.is_ok = False
                response.error_msg = f"Unsupported seg model: {self.seg_model_name=}"
                return response
            occupied_mask = (self.voxel_map.voxels_flags == 1)
            combined_mask = occupied_mask & mask
            pos_2d = (self.voxel_map.grid_pos[combined_mask])[:, :2]
            if len(pos_2d) == 0:
                response.is_ok = False
                response.error_msg = f"[query_info_callback] Unable to find points on map corrisponding to {string}"
                return response
            # Configure the clustering based on the object size based on the parameters dict
            params = self.clustering_params.get(size, self.clustering_params["default"])
            dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            # Cluster the points with labels
            dbscan_labels = dbscan.fit_predict(pos_2d)
            # Publish the cluster as markers
            cluster_centers, cluster_sizes, hulls_verticies = compute_cluster_centers(np.array(pos_2d, dtype=int), dbscan_labels, self.voxel_map.cell_size, self.voxel_map.grid_size)
            marker_iter_counter += 1
            # Publish Marker as polygons for the bounding contours
            if hulls_verticies is not None and hulls_verticies != []:
                pub_clusters_verticies(hulls_verticies, self.map_frame, self.get_clock().now().to_msg(), self.bbox_marker_pub)
            # Covert python array to ros2 msg
            objects = []
            for e, pose in enumerate(cluster_centers):
                pose = pose * self.voxel_map.cell_size - (self.voxel_map.grid_size * self.voxel_map.cell_size / 2)
                pose = np.append(pose, 0.0)
                pose_mapframe = self.rot_mat_z @ pose.T
                vox_object_msg = VoxelObject()
                vox_object_msg.pose.append(pose_mapframe[0])
                vox_object_msg.pose.append(pose_mapframe[1])
                objects.append(vox_object_msg)
            obj = ObjectQueryResult()
            obj.objects = objects
            obj.object_string = string
            response.objects_poses.append(obj)
        
        # Compute orientation around Z: theta
        q = robot_pose.orientation
        theta =  np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)) 
        response.robot_pose = np.array([robot_pose.position.x, 
                                        robot_pose.position.y,
                                        theta])
        response.is_ok = True
        response.error_msg = ''
        self.get_logger().info(f"[query_info_callback] Sending back response with objects at: {response.objects_poses}\n\n")
        self.get_logger().info(f"Callback time {time.time() - start}")
        return response

    def index_map_callback(self, request : IndexMap.Request, response : IndexMap.Response):
        if not self.initialized:
            response.is_ok = False
            response.error_msg = f"Node not initialized. Call the method intialize() first."
            return response
        try:
            self.get_logger().info("[index_map_callback] start callback")
            start=time.time()
            if not self.tf_published:
                publish_static_transform(stamp=self.get_clock().now().to_msg(),
                                         frame_id= self.map_frame,
                                         child_frame_id=self.semantic_map_frame_name,
                                         grid_size=self.voxel_map.grid_size,
                                         cell_size=self.voxel_map.cell_size,
                                         tf_static_broadcaster=self.tf_static_broadcaster)
            # Index the map and find the unique points on the 2D plane that match the indexing string
            if self.seg_model_name == "fcclip":
                mask = self.voxel_map.index_map_fcclip(request.indexing_string,
                                                       fcclip_m2f_cos_sim_threshold=self.fcclip_m2f_cos_sim_threshold,
                                                       fcclip_clip_cos_sim_threshold=self.fcclip_clip_cos_sim_threshold,
                                                       batch_size=self.batch_size
                                                )
            elif self.seg_model_name == "lseg":
                mask = self.voxel_map.index_map_lseg(language_desc=request.indexing_string, 
                                            clip_model=self.lseg_clip_model, 
                                            lseg_clip_feat_dim=self.lseg_clip_feat_dim, 
                                            lseg_use_cosine_sim=self.lseg_use_cosine_sim, 
                                            lseg_clip_threshold=self.lseg_cossim_threshold,
                                            batch_size=self.batch_size
                                            )
            else:
                self.get_logger().error(f"[index_map_callback] Unsupported seg model: {self.seg_model_name=}")
                response.is_ok = False
                response.error_msg = f"Unsupported seg model: {self.seg_model_name=}"
                return response
            occupied_mask = (self.voxel_map.voxels_flags == 1)
            # Publish the results as pointcloud (for visualization)
            grid_rgb = self.voxel_map.grid_rgb.copy()
            grid_rgb[:] = [0, 0, 255]
            grid_rgb[mask] = [255, 0, 0]
            color = grid_rgb[occupied_mask]
            points = self.voxel_map.grid_pos[occupied_mask] * self.voxel_map.cell_size  #scale it to meters
            msg = xyzrgb_array_to_pointcloud2(points, color, stamp=self.get_clock().now().to_msg(), frame_id=self.semantic_map_frame_name)
            self.index_pointcloud_pub.publish(msg)
            self.get_logger().info(f"Callback time: {time.time() - start}")
            if len(grid_rgb[mask]) == 0:
                response.is_ok = False
                response.error_msg = f"[index_map_callback] Unable to find points on map corrisponding to {request.indexing_string}"
                return response
            response.is_ok = True
            return response

        except Exception as ex:
            print(f"Unexpected exception: {ex=}, {type(ex)=}")
            response.is_ok = False
            response.error_msg = f"[index_map_callback] An exception occurred:{ex=}"
            return response

    def pub_goal_callback(self, request : PublishGoal.Request, response : PublishGoal.Response):
        if not self.initialized:
            response.is_ok = False
            response.error_msg = f"Node not initialized. Call the method intialize() first."
            return response
        try:
            if not self.costmap_ready:
                response.is_ok = False
                response.error_msg = f"[pub_goal_callback] Costmap not available. Check that the topic {self.get_parameter('costmap_topic_name').value} is available."
                self.get_logger().error(response.error_msg)
                return response
            if len(request.goal_pose) < 3:
                response.is_ok = False
                response.error_msg = f"[pub_goal_callback] Received goal pose with {len(request.goal_pose)=}. x, y, theta are needed for publishing a goal."
                self.get_logger().error(response.error_msg)

            # Convert goal pose in grid cell
            map_coords_x, map_coords_y = convert_in_costmap_cell(request.goal_pose[0], request.goal_pose[1], self.global_costmap_msg)
            # Find the voxels near the goal
            goal_voxel = to_grid_coords(request.goal_pose[0], request.goal_pose[1], 0.0, self.voxel_map.grid_size, self.voxel_map.cell_size)

            # Check grid bounds
            min_x = goal_voxel[0] - self.costmap_goal_window
            if min_x < 0 : min_x = 0
            min_y = goal_voxel[1] - self.costmap_goal_window
            if min_y < 0 : min_y = 0

            max_x = goal_voxel[0] + self.costmap_goal_window
            if max_x > self.voxel_map.grid_size : max_x = self.voxel_map.grid_size
            max_y = goal_voxel[1] + self.costmap_goal_window
            if max_y > self.voxel_map.grid_size : max_y = self.voxel_map.grid_size

            occ_mask = self.voxel_map.occupied_ids[min_x:max_x, min_y:max_y] >= 0
            ids = self.voxel_map.occupied_ids[min_x:max_x, min_y:max_y][occ_mask]
            seen_voxels = self.voxel_map.grid_pos[ids]
            # I have to eliminate the voxels which have been removed/unmarked by raycasting
            valid_mask = self.voxel_map.voxels_flags[ids] == 1
            near_voxels = seen_voxels[valid_mask]
            near_voxels_meters = np.zeros_like(near_voxels, np.float32)
            costmap = copy.deepcopy(self.costmap_grid_bool)
            # Convert voxels into global map frame
            if len(near_voxels) > 0:
                for i in range(0, len(near_voxels)):
                    near_voxels_meters[i] = from_grid_coords(near_voxels[i][0], near_voxels[i][1], near_voxels[i][2], self.voxel_map.grid_size, self.voxel_map.cell_size)
                # Remove voxels on the ground
                ground_mask = (near_voxels_meters > self.costmap_floor_height)[:, 2]
                # Project to 2D and find unique points
                obstacle_points = np.unique((near_voxels_meters[ground_mask])[:, 0:2], axis=0)
                # Convert to costmap cells and get cells boundaries
                cell_min_x = 0
                cell_min_y = 0
                cell_max_x = 0
                cell_max_y = 0
                costmap_cells = np.zeros_like(obstacle_points, np.int32)
                for k in range(0, len(obstacle_points)):
                    costmap_cells[k] = convert_in_costmap_cell(obstacle_points[k][0], obstacle_points[k][1], self.global_costmap_msg)
                    if k == 0:
                        cell_min_x = costmap_cells[k][0]
                        cell_min_y = costmap_cells[k][1]
                        cell_max_x = costmap_cells[k][0]
                        cell_max_y = costmap_cells[k][1]
                    else:   #find min and max
                        if costmap_cells[k][0] < cell_min_x : cell_min_x = costmap_cells[k][0]
                        elif costmap_cells[k][0] > cell_max_x : cell_max_x = costmap_cells[k][0]
                        if costmap_cells[k][1] < cell_min_y : cell_min_y = costmap_cells[k][1]
                        elif costmap_cells[k][1] > cell_max_y : cell_max_y = costmap_cells[k][1]

                l_bound_x = cell_min_x - self.costmap_dilation
                u_bound_x = cell_max_x + self.costmap_dilation
                l_bound_y = cell_min_y - self.costmap_dilation
                u_bound_y = cell_max_y + self.costmap_dilation
                # Check for out of bound
                if l_bound_x < 0 : l_bound_x = 0
                if u_bound_x > costmap.shape[0] : u_bound_x = costmap.shape[0]
                if l_bound_y < 0 : l_bound_y = 0
                if u_bound_y > costmap.shape[1] : u_bound_y = costmap.shape[1]
                
                # Get only the part of the image where we are adding points
                binary_img = np.full_like(costmap, False)
                binary_img[costmap_cells] = True
                binary_img = binary_img[l_bound_x:u_bound_x, l_bound_y:u_bound_y]
                # Dilate
                cv_img = binary_img.astype(np.uint8)
                kernel = np.ones((2*self.costmap_dilation, 2*self.costmap_dilation), np.uint8)
                dilated_img = np.array(cv2.dilate(cv_img, kernel, iterations = 1), dtype=bool)
                # Add to costmap
                costmap[l_bound_x:u_bound_x, l_bound_y:u_bound_y] = costmap[l_bound_x:u_bound_x, l_bound_y:u_bound_y] | dilated_img
                # Debug
                #self.pub_costmap_markers(costmap)


            # Now find the closest free cell on the costmap
            free_cell = find_closest_free_cell(costmap, map_coords_x, map_coords_y)
            goal_pose_x, goal_pose_y = convert_from_costmap_cell(free_cell[0], free_cell[1], self.global_costmap_msg)
            goal_pose_theta = request.goal_pose[2]  # TODO for the moment we keep the same orientation, should we do any averaging?
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = self.map_frame
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.position.x = goal_pose_x
            goal_msg.pose.position.y = goal_pose_y
            goal_msg.pose.position.z = 0.0
            q = quaternion_from_euler(0.0, 0.0, goal_pose_theta)
            goal_msg.pose.orientation.x = q[0]
            goal_msg.pose.orientation.y = q[1]
            goal_msg.pose.orientation.z = q[2]
            goal_msg.pose.orientation.w = q[3]
            self.goal_pub.publish(goal_msg)
            response.is_ok = True
        except Exception as ex:
            response.is_ok = False
            response.error_msg = f"[pub_goal_callback] {ex=}"
            print(response.error_msg)
        
        return response

    def show_rgb_map_callback(self, request : ShowMap.Request, response : ShowMap.Response):
        if not self.initialized:
            response.is_ok = False
            response.error_msg = f"Node not initialized. Call the method intialize() first."
            return response
        try:
            if not self.voxel_map.map_present:
                response.is_ok = False
                response.error_msg = f"[show_rgb_map_callback] Map not loaded. Call load_map service before with a valid path."
                return response
            if not self.tf_published:
                publish_static_transform(stamp=self.get_clock().now().to_msg(),
                                         frame_id= self.map_frame,
                                         child_frame_id=self.semantic_map_frame_name,
                                         grid_size=self.voxel_map.grid_size,
                                         cell_size=self.voxel_map.cell_size,
                                         tf_static_broadcaster=self.tf_static_broadcaster)
            occupied_mask = (self.voxel_map.voxels_flags == 1) 
            pos_mask = (self.voxel_map.grid_pos[occupied_mask] > 0).all(axis=1)
            color = self.voxel_map.grid_rgb[occupied_mask]
            positive_color = color[pos_mask]
            points = self.voxel_map.grid_pos[occupied_mask][pos_mask] * self.voxel_map.cell_size
            msg = xyzrgb_array_to_pointcloud2(points, positive_color, stamp=self.get_clock().now().to_msg(), frame_id=self.semantic_map_frame_name)
            self.map_pointcloud_pub.publish(msg)
        except Exception as ex:
            print(f"Unexpected exception: {ex=}, {type(ex)=}")
            response.is_ok = False
            response.error_msg = f"[show_rgb_map_callback] An exception occurred: {ex}"
            return response
        response.is_ok = True
        return response

    def load_map_callback(self, request : LoadMap.Request, response : LoadMap.Response):
        """
        Loads the map expressed in the path
        """
        if not self.initialized:
            response.is_ok = False
            response.error_msg = f"Node not initialized. Call the method intialize() first."
            return response
        try:
            # Check if map extension already present
            if ".h5df" in request.path:
                path = request.path
            else:
                path = request.path + ".h5df"
            if not self.voxel_map.load_map(path):
                response.is_ok = False
                response.error_msg=f"VLMap path not valid: the file {path} doesn't exist"
                return response
            else:
                # Show also the map
                showmap_req = ShowMap.Request()
                showmap_resp = ShowMap.Response()
                _ = self.show_rgb_map_callback(showmap_req, showmap_resp)
                response.is_ok = True
                #self.map_loaded = True
                return response
        except:
            print("[load_map_callback] An exception occurred")
            response.is_ok = False
            response.error_msg="An exception occurred"
            return response
    
    def costmap_callback(self, msg : OccupancyGrid):
        """
        Stores the global costmap msg in self.global_costmap_msg
        """
        self.global_costmap_msg = msg
        self.costmap_grid = np.array(self.global_costmap_msg.data).reshape((self.global_costmap_msg.info.height, self.global_costmap_msg.info.width)).T
        self.costmap_grid_bool = (self.costmap_grid > self.occupied_threshold)
        self.costmap_ready = True

    def eval_callback(self, request : EvaluateMap.Request, response : EvaluateMap.Response):
        """
        Function that provides the interest point of a given textual query.
        Used by the evaluation script only.
        """
        try:
            self.get_logger().info(f"[eval_callback] Got request {request.indexing_string}")
            # For visualization purposes only
            ind_req = IndexMap.Request()
            ind_req.indexing_string = request.indexing_string
            ind_resp = IndexMap.Response()
            _ = self.index_map_callback(ind_req, ind_resp)
            # Get thevalues of the points matching the request
            if self.seg_model_name == "fcclip":
                mask = self.voxel_map.index_map_fcclip(request.indexing_string,
                                                       fcclip_m2f_cos_sim_threshold=self.fcclip_m2f_cos_sim_threshold,
                                                       fcclip_clip_cos_sim_threshold=self.fcclip_clip_cos_sim_threshold,
                                                       batch_size=self.batch_size
                                                )
            elif self.seg_model_name == "lseg":
                mask = self.voxel_map.index_map_lseg(language_desc=request.indexing_string, 
                                                clip_model=self.lseg_clip_model, 
                                                lseg_clip_feat_dim=self.lseg_clip_feat_dim, 
                                                lseg_use_cosine_sim=self.lseg_use_cosine_sim, 
                                                lseg_clip_threshold=self.lseg_cossim_threshold,
                                                batch_size=self.batch_size
                                                )
            else:
                self.get_logger().error(f"[eval_callback] Unsupported seg model: {self.seg_model_name=}")
                response.is_ok = False
                response.error_msg = f"Unsupported seg model: {self.seg_model_name=}"
                return response
            occupied_mask = (self.voxel_map.voxels_flags == 1)
            # Valid map points
            map_pc = self.voxel_map.grid_pos[occupied_mask]
            response.map_pc = []
            for map_point in map_pc:
                p = Point()
                p.x = float(map_point[0])
                p.y = float(map_point[1])
                p.z = float(map_point[2])
                response.map_pc.append(p)
            # Give the indexed points in output
            points = self.voxel_map.grid_pos[mask & occupied_mask]
            if len(points) > 0 and points is not None:
                response.pointcloud = []
                for point in points:
                    p = Point()
                    p.x = float(point[0])
                    p.y = float(point[1])
                    p.z = float(point[2])
                    response.pointcloud.append(p)
                response.is_ok= True
                return response
            else:
                response.pointcloud = []
                response.is_ok = False
                response.error_msg = f"No points found for: {request.indexing_string}"
                return response
        except Exception as ex:
            self.get_logger().error(f"[eval_callback] Caught exception {ex}")
            response.is_ok = False
            response.error_msg = f"Caught exception {ex}"
            return response

def main(args=None):
    rclpy.init(args=args)
    ros_node = SemanticMapServer("voxmap_srv_node")
    # to manually set the parameter for the seg_model_name
    seg_mod_name = ros_node.get_parameter('seg_model_name')
    
    if seg_mod_name.value == "fcclip":
        print("FCCLIP\n\n\n")
        fcclip_init_mod = importlib.import_module("visual_language_navigation.utils.fcclip_utils")
        init_fcclip = fcclip_init_mod._init_fcclip
        lseg_model, _ = init_fcclip()
    else:
        # go by default
        lseg_model = None
    ros_node.initialize(lseg_model)
    rclpy.spin(ros_node)

    rclpy.shutdown()

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
