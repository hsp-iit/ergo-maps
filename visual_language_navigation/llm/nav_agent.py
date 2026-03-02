# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import rclpy
import rclpy.duration
import rclpy.executors
from rclpy.node import Node
from ros2_vlmaps_interfaces.srv import IndexMap, LlmQuery, LlmChat, PublishGoal
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import re
import os
import json
import rclpy.time
from rclpy.time import Duration

import numpy as np
from typing import Tuple
from openai import AzureOpenAI
import openai

import llm_tools
from visual_language_navigation.utils.conversion_utils import quaternion_from_euler

def load_config(file_path):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Remove any leading/trailing whitespace, including newline characters
            line = line.strip()
            if line and '=' in line:
                # Split the line into key and value
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

class LLMWrapper(Node):
    def __init__(self, 
                 node_name : str
                 ) -> None:
        super().__init__(node_name)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_query_service', "semantic_map_server/llm_query"),
                ('show_index_map_service', "semantic_map_server/index_map"),
                ('agent_chat_service', "agent_comm"),
                ('use_remote_pose', True),
                ('config_path', '/home/user1/visual-language-navigation/config.env'),
                ('goal_pub_service', 'pub_goal'),
                ('refresh_conversation', True),
                ('use_chatGPT', True),
                ('model_name', "hsp-Vocalinteraction_gpt4o"),
                ('robot_base_link', "geometric_unicycle"),
                ('robot_name', "ergoCub")
            ])
        openai_config = load_config(self.get_parameter('config_path').value)
        map_service_name = self.get_parameter('map_query_service').value
        show_index_map_service_name = self.get_parameter('show_index_map_service').value
        agent_chat_service = self.get_parameter('agent_chat_service').value
        self.use_remote_pose = self.get_parameter('use_remote_pose').value
        goal_pub_service_name = self.get_parameter('goal_pub_service').value
        self.refresh_conversation = self.get_parameter('refresh_conversation').value
        self.use_chatGPT = self.get_parameter('use_chatGPT').value
        self.model_name = self.get_parameter('model_name').value
        self.robot_name = self.get_parameter('robot_name').value
        self.base_frame = self.get_parameter('robot_base_link').value
        self.get_logger().info(f"Got robot name: {self.robot_name}")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_callback_group = ReentrantCallbackGroup() 
        self.tf_timer = self.create_timer(0.2, self.tf_callback, callback_group=self.tf_callback_group)
        
        self.client = openai
        if self.use_chatGPT:
            self.client = AzureOpenAI(
                azure_endpoint=f"{openai_config['AZURE_ENDPOINT']}", #do not add "/openai" at the end here because this will be automatically added by this SDK
                api_key=openai_config['AZURE_API_KEY'],
                api_version="2024-10-21"
            )
        else:
            openai.base_url = "http://localhost:11434/v1/"
            openai.api_key = 'ollama'
            
        self.set_conversation()
        self.map_callback_group = ReentrantCallbackGroup()
        self.map_client = self.create_client(LlmQuery, map_service_name, callback_group=self.map_callback_group)
        self.show_index_map_callback_group = ReentrantCallbackGroup() 
        self.show_index_map_client = self.create_client(IndexMap, show_index_map_service_name, callback_group=self.show_index_map_callback_group)
        # Wait for it to become available
        while not self.map_client.service_is_ready():
            self.get_logger().warn(f"[{self.get_name()}] Waiting for service {map_service_name} to become available")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))

        # Goal pub service
        self.goal_callback_group = ReentrantCallbackGroup()
        self.goal_client = self.create_client(PublishGoal, goal_pub_service_name, callback_group=self.goal_callback_group)
        while not self.goal_client.service_is_ready():
            self.get_logger().warn(f"[{self.get_name()}] Waiting for service {goal_pub_service_name} to become available")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))

        self.ll_text_prompt = self.create_service(LlmChat, agent_chat_service, self.chat_callback)

        # For llm reasoning publishing
        self.thought_pub = self.create_publisher(String, "llm_reasoning", 10)

        self.robot_pose = None
        self.latest_used_robot_pose = [0.0, 0.0, 0.0]
        self.tf_robot_pose = [0.0, 0.0, 0.0]
        self.llm_marker_pub = self.create_publisher(Marker, "/llm_marker", 10)

        self.available_functions = {
            'get_environment_info': self.get_environment_info,
            'show_indexed_map': self.show_indexed_map
        }
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_environment_info",
                    "description": """
            Get the information about the environment. It returns the robot pose and size of all the objects instances in the environment.
            :return: a string containing the robot pose as first element, expressed as a vector (x, y, theta), and a list of 2D poses (x, y) of all the found instances of the object in the environment.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_string": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "a list of each string of the objects to search in the environment.",
                            },
                            "object_size": {
                                "type": "string",
                                "description": "each numerical size category of the objects to search in the environment. Separated by a blank space ' '.",
                            },
                        },
                        "required": ["object_string", "object_size"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "show_indexed_map",
                    "description": """
                        Show category in voxel map on rviz, red for target, blue otherwies

                        :param category: each string of the objects to search in the environment. Separated by a blank space ' '.
                        """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Name of category you want to show.",
                            }
                        },
                        "required": ["category"],
                    }
                }
            }
        ]
            
    def set_conversation(self) :
        self.system_instructions = """
        You are a robot navigating in a 2D Cartesian plane. Follow user instructions to determine a goal pose in your environment.

        ENVIRONMENT:
        - Your reference frame is 'XYZ_robot':
            X positive = forward, X negative = backward
            Y positive = left, Y negative = right
            Pose = (x_robot, y_robot, theta) in meters/radians
        - All object positions are expressed relative to your reference frame
        - Your circular footprint radius = 0.5m

        OBJECT RULES:
        - Always call the tool 'get_environment_info' to retrieve objects before reasoning
        - Assign size categories to objects:
            0 = very small (bottle, mug, lamp, pen)
            1 = small/medium (chair, stool, box, laptop, bag)
            2 = big (table, door, window)
            3 = very big (wall, shelving, wardrobe)
        - When multiple objects are requested, query all at once
        - Include **object qualities/attributes** when querying, e.g., "orange cup", "blue bottle"
        - When a user describes an object with additional attributes or related objects (e.g., "something with other"), you must:
            1. Treat each noun as a separate object to search for (here: "something" and "other").
            2. Call get_environment_info separately for each object type.
            3. Reason about their spatial relationship (e.g., find the "something" closest to the "other").
        - Apply reasoning to all instances of objects, not just some
        - Consider footprint when approaching objects; stop at offset from object

        ORIENTATION RULES:
        - If orientation is explicitly given, use that value.
        - If orientation is not specified, you MUST face the object or goal position.
            - "Face the object" means that you have to compute the orientation as the angle from the final robot position (x_goal, y_goal) to the object (x_object, y_object):
                theta = atan2(y_object - y_goal, x_object - x_goal)
            - Compute theta as the angle of this line in radians.

        OUTPUT RULES:
        - FINAL RESPONSE MUST END WITH ONE LINE AND BE EXACTLY THREE NUMBERS: (x, y, theta)
        - Output only numbers; do not use π, θ, functions, or formatting symbols
        - If not clearly specified, face the target object/position
        - Objects are IN FRONT if X>0 and Y in [-2, 2]; BEHIND if X<0 and Y in [-2, 2]
        - If asked to stop: respond (0, 0, 0)
        - If asked to go to tools, pass 'tools' as argument to tool call
        """
        
        #RELATION RULES:
        #- When the user specifies an object (A) with another object (B) on/with/near it (e.g., "table with laptop"), do the following:
        #    1. Query all instances of the object A (here: "table") and the other object B (here: "laptop") separately using get_environment_info.
        #    2. Determine spatial relations between objects A and B:
        #        - "with/on": The object A must be within 1.0m of the object B pose.
        #        - "near": The object A must be within 3.0m of the object B pose.
        #        - "near": The object A must be the closest one to the object B pose.
        #        - Always check **all combinations** of the two object types to ensure no valid pair is missed. Evaluate all the combinations and consider all of them to be as valid candidates.
        #    3. If multiple candidates remain, compute the distance from the robot to each candidate.
        #- Once the best candidate is found, compute the goal pose according to OBJECT and ORIENTATION RULES.

        self.conversation_flow = [{'role': 'system', 'content': self.system_instructions}]
        
    def chat_callback(self, request : LlmChat.Request, response : LlmChat.Response) :
        prompt = request.text
        # Add the prompt to the flow of messages
        self.conversation_flow.append({'role': 'user', 'content': prompt})
        llm_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_flow,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = llm_response.choices[0].message
        self.conversation_flow.append(response_message)
        if response_message.content != None:
            self.thought_flow_pub(response_message.content)

        print(f"Model's response: {response_message}")

        # Reasoning loop
        count = 0
        got_pose = False    # TODO improve
        while response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function call: {function_name}")  
                print(f"Function arguments: {function_args}")  

                if function_call := self.available_functions.get(function_name):
                    function_response = function_call(**function_args)
                    print(f"Function result: {function_response}") 
                    if function_response == None:
                        function_response = "Error in tool call"
                    self.conversation_flow.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                    got_pose = True
                    count+=1
                else:
                    function_response = json.dumps({"error": "Unknown function"})
            
            llm_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_flow,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1
            )
            response_message = llm_response.choices[0].message
            self.thought_flow_pub(response_message.content)
            self.conversation_flow.append(response_message)
        
        final_response = llm_response
        print('Final response:', final_response.choices[0].message.content)
        self.thought_flow_pub(final_response.choices[0].message.content)
        try:
            if (final_response.choices[0].message.content == "") :
                last_line = self.conversation_flow[-1].message.content.strip().split('\n')[-1]  # TODO test
            else:
                text_lines = final_response.choices[0].message.content.strip().split('\n')
                last_line = text_lines[-1]
                # Search also the second last line if the llm does a wrong formatting
                if len(text_lines) >=2 :
                    second_last_line = text_lines[len(text_lines) - 2]
                else:
                    second_last_line = None

            pattern = r"\(?\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*\)?"

            # Maintain context and save messages
            if final_response.choices[0].message.content != "":
                self.conversation_flow.append({'role': 'assistant', 'content': last_line})

            # Search the last line for something TODO search also the second last
            pose_found = re.search(pattern, last_line)
            if (pose_found):
                match = pose_found
            else:
                print("No numbers found in the last line of the LLM response... trying the second last line:")
                # Search the second last if nothing has been found
                match = re.search(pattern, second_last_line)

            # Convert string to array
            if match:
                x, y, z = map(float, (match.group(1), match.group(3), match.group(5)))  #z is theta
                goal = np.array([x, y, z])
                print(f"Goal in robot frame: {goal}")
                # Used only for relative movement instructions -i.e. turn around
                if not got_pose:
                    self.latest_used_robot_pose = self.tf_robot_pose

                print(f"Robot pose: {self.latest_used_robot_pose}")
                _, goal_map_frame = llm_tools.transform_from_robot_frame(self.latest_used_robot_pose, "tmp", goal)
                print(f"Goal in map frame: {goal_map_frame}")
                self.publish_llm_markers(goal_map_frame)
                # Publish Goal
                goal_request = PublishGoal.Request()
                goal_request.goal_pose.append(goal_map_frame[0][0])
                goal_request.goal_pose.append(goal_map_frame[0][1])
                goal_request.goal_pose.append(goal_map_frame[0][2])
                self.future = self.goal_client.call_async(request=goal_request)
                print("Spinning goal client request")
                self.executor.spin_until_future_complete(self.future)    # TODO add timeout and handle exception
                goal_response : PublishGoal.Response = self.future.result()
                print(f"{goal_response.is_ok=} {goal_response.error_msg}")
                response.is_ok = True
                response.error_msg = ''
            else:
                print("No numbers found in the last two lines of the LLM response.")
                response.is_ok = False
                response.error_msg = 'No numbers found in the last two lines of the LLM response'
            
        except Exception as ex:
            print(f"{ex=}")
            response.is_ok = False
            response.error_msg = f"{ex=}"
        
        if self.refresh_conversation:
            self.set_conversation()
            
        return response

    def show_indexed_map(self, category: str):
        """
        Show category in voxel map on rviz, red for target blue otherwies

        :param category: each string of the objects to search in the environment. Separated by a blank space ' '.
        """
        request = IndexMap.Request()
        request.indexing_string = category
        print(request)
        # Need another executor because the main thread is locked by this callback.
        # So to use the future, it's necessary to have another thread to spin the node
        self.future = self.show_index_map_client.call_async(request=request)
        self.executor.spin_until_future_complete(self.future)
        
        response : IndexMap.Response = self.future.result()
        if response.is_ok:
            return "Results shown to user"

    def get_environment_info(self, object_string, object_size: str) -> Tuple:
        """
        Get the information about the environment. It returns the robot pose and all the objects instances in the environment.

        :param object_string: each string of the objects to search in the environment. Separated by a blank space ' '.
        :param object_size: each size category of the objects to search in the environment. Separated by a blank space ' '.
        :return: a string containing list of 2D poses (x, y) of all the found instances of the required object. Expressed in the robot perspective.
        """
        request = LlmQuery.Request()
        request.object_string = object_string
        request.object_size = object_size
        # Need another executor because the main thread is locked by this callback.
        # So to use the future, it's necessary to have another thread to spin the node
        self.future = self.map_client.call_async(request=request)
        self.executor.spin_until_future_complete(self.future)    # TODO add timeout and handle exception
        response : LlmQuery.Response = self.future.result()

        if response.is_ok:
            # Convert data in a more readable format for the LLM
            self.latest_used_robot_pose = response.robot_pose

            results = []
            for i in range(len(response.objects_poses)):
                # Compose the string by adding the string of the object first
                object_string = response.objects_poses[i].object_string
                obj_list = []
                #size_list = []
                for obj in response.objects_poses[i].objects:
                    obj_list.append(list(obj.pose))
                    #size_list.append(obj.size)
                _, obj_list_robot_frame = llm_tools.transform_in_robot_frame(self.latest_used_robot_pose, object_string, obj_list)
                results.append({"object_type": object_string, "objects": [{"pose": x[0]} for x in zip(obj_list_robot_frame)]})
            json_result_array = json.dumps(results, indent=4)
            json_answer = "object information: " + json_result_array
                
            return json_answer
        else:
            return None


    def publish_llm_markers(self, positions_list):
        marker = Marker()
        marker.id = 0
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.3
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.frame_locked = True
        c = ColorRGBA()
        c.r, c.b, c.g  = np.array([1.0, 0.0, 0.0])  # In red we publish the marker in the center of the cluster
        c.a = 1.0
        marker.color = c
        np_positions_list = np.array(positions_list)
        if np_positions_list.ndim > 1: 
            if len(np_positions_list) == 1: # Only one goal
                marker.pose.position.x = np_positions_list[0][0]
                marker.pose.position.y = np_positions_list[0][1]
                # Convert orientation
                quat = quaternion_from_euler(0.0, 0.0, np_positions_list[0][2])
                marker.pose.orientation.x = quat[0]
                marker.pose.orientation.y = quat[1]
                marker.pose.orientation.z = quat[2]
                marker.pose.orientation.w = quat[3]
                p = Point()
                p.x = 0.0
                p.y = 0.0
                p.z = 0.0 
                marker.points.append(p)
                p2 = Point()
                p2.x = 0.3
                p2.y = 0.0
                p2.z = 0.0 
                marker.points.append(p2)

            else:    # TODO update to marker array list
                for item in positions_list:
                    p = Point()
                    p.x = item[0]
                    p.y = item[1]
                    p.z = 0.20  # height
                    marker.points.append(p)
                    c = ColorRGBA()
                    c.r, c.b, c.g  = np.array([1.0, 0.0, 0.0])  # In red we publish the marker in the center of the cluster
                    c.a = 0.9
                    marker.colors.append(c)
        else:   # Only one marker passed as a single array
            p = Point()
            marker.pose.position.x = np_positions_list[0]
            marker.pose.position.y = np_positions_list[1]
            # Convert orientation
            quat = quaternion_from_euler(0.0, 0.0, np_positions_list[2])
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            p.x = 0.0
            p.y = 0.0
            p.z = 0.0 
            marker.points.append(p)
            p2 = Point()
            p2.x = 0.1
            p2.y = 0.0
            p2.z = 0.0 
            marker.points.append(p2)
        
        self.llm_marker_pub.publish(marker)
    
    def tf_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                    "map",              
                    self.base_frame,  
                    rclpy.time.Time(),
                    timeout=Duration(seconds=3.0)
                    )
            self.tf_robot_pose[0] = transform.transform.translation.x
            self.tf_robot_pose[1] = transform.transform.translation.y
            q = transform.transform.rotation
            self.tf_robot_pose[2] =  np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)) 
        except Exception as ex:
            self.get_logger().warn(f"[query_info_callback] Robot pose not available, {ex=}")
            return
    
    def thought_flow_pub(self, text : str, is_robot = True):
        msg = String()
        if text is None:
            return
        if is_robot:
            msg.data = "robot: " + text
        else:
            msg.data = "user: " + text
        self.thought_pub.publish(msg)

def main(args=None):
    rclpy.init()
    node = LLMWrapper("llm_wrapper_node")
    print(f"Created node: {node.get_name()}")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    print("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()
    return

if __name__=="__main__":
    main()
