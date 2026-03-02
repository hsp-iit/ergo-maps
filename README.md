# ERGO-Maps: Embodied Representations for Global Occupancy Maps

This repository provides a ROS 2-based framework for **semantic navigation using visual-language models**.  
It integrates segmentation models like [**LSeg**](https://github.com/isl-org/lang-seg) and [**FC-CLIP**](https://github.com/bytedance/fc-clip) within a Dockerized environment for reproducibility and ease of setup.

---

## :whale: Docker Setup

### Requirements

- **GPU** with support for `nvidia/cuda:12.8.0`
- **At least 8 GB of VRAM** (can be reduced by lowering the batch size)
- [Docker](https://docs.docker.com/engine/install/ubuntu)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

### 1. Download Model Checkpoints

Before building the Docker image, download the checkpoints for the segmentation models.

From the `docker` directory, run:

```bash
python3 download_checkpoint.py <PATH_TO_THIS_REPO>/docker
python3 download_cocopan_checkpoint.py
```

These scripts will automatically fetch the necessary pretrained weights for **LSeg** and **FC-CLIP**.

---

### 2. Build the Docker Image

Inside `build_docker.sh`, set your preferred image name after the `-t` flag.  
Then build the image using:

```bash
. build_docker.sh GITHUB_USERNAME GITHUB_EMAIL GITHUB_TOKEN
```

> **Note:** The GitHub credentials are required to authenticate and access this repository during the build process.

---

### 3. Run the Docker Container

After a successful build, edit `run.sh` to match the image name and tag used during the build, then run:

```bash
. run.sh
```

---

## :rocket: Running the Code

### 1. Workspace Preparation

This repository is a **ROS 2 package**.  
To use ROS 2 launch files and parameters, build the workspace with **colcon**:

```bash
colcon build --symlink-install
source install/local_setup.bash
```

Additionally, compile and source the ROS 2 message interface:

```bash
cd ros2_vlmaps_interfaces
colcon build --symlink-install
source install/local_setup.bash
```

> You can add the `source` command to your `~/.bashrc` for convenience.

---

### 2. Launch Files

#### Mapping Mode

```bash
ros2 launch visual_language_navigation build_map.launch.py
```

#### Query Existing Map

```bash
ros2 launch visual_language_navigation semantic_map_server.launch.py
```

#### Navigation with LLM Agents

If you already have a semantic map, launch the `semantic_map_server`.  
If you need to build or update one, use the `semantic_map_builder`.  
Then run the following nodes:

```bash
python3 visual_language_navigation/llm/nav_agent.py
python3 visual_language_navigation/llm/chat_agent.py
```

---

## :telescope: ROS 2 Topics

### Inputs

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/rgbd/img` | [`sensor_msgs/Image`](https://docs.ros2.org/foxy/api/sensor_msgs/msg/Image.html) | RGB image from the camera |
| `/camera/rgbd/depth` | [`sensor_msgs/Image`](https://docs.ros2.org/foxy/api/sensor_msgs/msg/Image.html) | Depth image from the camera |
| `/amcl_pose` | [`geometry_msgs/PoseWithCovariance`](https://docs.ros2.org/foxy/api/geometry_msgs/msg/PoseWithCovariance.html) | Pose estimate from [Nav2 AMCL](https://docs.nav2.org/configuration/packages/configuring-amcl.html) |
| `/global_costmap/costmap` | [Nav2 Costmap](https://api.nav2.org/msgs/jazzy/costmap.html) | Occupancy grid map |
| `/camera/rgbd/camera_info` | [`sensor_msgs/CameraInfo`](https://docs.ros2.org/foxy/api/sensor_msgs/msg/CameraInfo.html) | Camera intrinsics (optional, can be manually set in `mapping_params.yaml`) |

### Outputs

| Topic | Type | Description |
|-------|------|-------------|
| `/voxmap` | [`sensor_msgs/PointCloud2`](https://docs.ros2.org/foxy/api/sensor_msgs/msg/PointCloud2.html) | RGB-colored semantic map |
| `/map_index_result` | `PointCloud2` | Result of the `semantic_map_server/index_map` service |
| `/map_goal_indexing` | `PointCloud2` | Map goal indexing output |
| `/map_2d_index_marker` | `PointCloud2` | 2D index markers |

---

## :magic_wand: ROS 2 Services

| Service | Description |
|----------|--------------|
| `semantic_map_builder/enable_mapping` | Enables or disables the mapping callback (`enable_flag`: bool) |
| `semantic_map_server/index_map` | Searches the map for `indexing_string` and publishes the result on `/map_index_result` |
| `semantic_map_server/show_semantic_map` | Publishes the RGB semantic point cloud on `/voxmap` |
| `semantic_map_server/load_semantic_map` | Loads a semantic map from a specified `path` |
| `semantic_map_server/llm_query` | Retrieves information from the semantic map for the LLM agent |
| `chat_agent/user_text` | Sends user queries (transcribed text) to the LLM planner |

---

## :gear: Parameters

Main parameters are defined in [`params/mapping_params.yaml`](params/mapping_params.yaml).  
Key ones include:

| Parameter | Default | Description |
|------------|----------|-------------|
| `cell_size` | `0.02` | Cell size (m) |
| `grid_size` | `1500` | Number of cells per axis (map resolution) |
| `maximum_height` | `2.3` | Maximum voxel grid height (m) |
| `max_camera_distance` | `3.0` | Maximum Z-distance from the camera (m) |
| `depth_downsampling` | `10` | Random downsampling factor for RGB and depth pixels |
| `robot_base_frame` | `"geometric_unicycle"` | Robot base frame |
| `target_frame` | `"map"` | Global navigation frame |
| `map_frame` | `"map"` | Reference frame used by Nav2 |
| `seg_model_name` | `"fcclip"` | Segmentation model name |
| `classes_to_skip` | `["person", "floor", "ceiling", "wall"]` | Classes excluded from mapping |

---

## :brain: Notes

- This branch allows choosing between **LSeg** and **FC-CLIP** as the segmentation backend.  
- The Dockerfile provides a ready-to-use development environment for ease of setup.  
- Ensure your GPU drivers and CUDA runtime are correctly configured before running.


## Cite this work

TODO

### Other Citation
This work has taken inspiration from VLMaps: https://github.com/vlmaps/vlmaps

```bibtex
@inproceedings{huang23vlmaps,
               title={Visual Language Maps for Robot Navigation},
               author={Chenguang Huang and Oier Mees and Andy Zeng and Wolfram Burgard},
               booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
               year={2023},
               address = {London, UK}
} 
```
