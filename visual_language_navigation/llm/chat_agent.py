# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import rclpy
import rclpy.duration
import rclpy.executors
from rclpy.node import Node
from ros2_vlmaps_interfaces.srv import LlmChat, CameraRGB
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
import json
import cv2

import numpy as np
from typing import Tuple
from openai import AzureOpenAI

import base64

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

class LLMChatter(Node):
    def __init__(self, 
                 node_name : str
                 ) -> None:
        super().__init__(node_name)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('agent_comm_service', "agent_comm"),
                ('user_chat_service', "chat_agent/user_text"),
                ('config_path', '/home/user1/visual-language-navigation/config.env'),
                ('refresh_conversation', True), # With images it's likely to go over maximum context lenght
                ('image_srv_name', 'vlmap_builder/robot_camera_rgb'),
                ('robot_name', 'ergoCub'),
                ('use_camera', False)
            ])
        openai_config = load_config(self.get_parameter('config_path').value)
        agent_comm_service = self.get_parameter('agent_comm_service').value
        user_chat_service = self.get_parameter('user_chat_service').value
        self.refresh_conversation = self.get_parameter('refresh_conversation').value
        image_srv_name = self.get_parameter('image_srv_name').value
        self.robot_name = self.get_parameter('robot_name').value
        self.use_camera = self.get_parameter('use_camera').value
        
        self.client = AzureOpenAI(
            azure_endpoint=f"{openai_config['AZURE_ENDPOINT']}", #do not add "/openai" at the end here because this will be automatically added by this SDK
            api_key=openai_config['AZURE_API_KEY'],
            api_version="2024-10-21"
        )
    
        # Set the Chatting Agent system prompt
        self.set_conversation()
        # ROS2 service used by User
        self.user_chat_srv = self.create_service(LlmChat, user_chat_service, callback=self.user_chat_callback)
        # ROS2 service client for communicating with the Navigation Agent 
        self.nav_agent_callback_group = ReentrantCallbackGroup() 
        self.nav_agent_client = self.create_client(LlmChat, agent_comm_service, callback_group=self.nav_agent_callback_group)
        # Wait for it to become available
        while not self.nav_agent_client.service_is_ready():
            self.get_logger().warn(f"[{self.get_name()}] Waiting for service {agent_comm_service} to become available")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))
        
        # ROS2 service for rgb image
        if self.use_camera:
            self.img_callback_group = ReentrantCallbackGroup() 
            self.img_client = self.create_client(CameraRGB, image_srv_name, callback_group=self.img_callback_group)
        # Wait for it to become available
            while not self.img_client.service_is_ready():
                self.get_logger().warn(f"[{self.get_name()}] Waiting for service {image_srv_name} to become available")
                self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))
        
        # For showing the reasoning on a chat web page
        self.thought_pub = self.create_publisher(String, "llm_reasoning", 10)

        # Tool definitions used by LLM
        self.available_functions = {
            'query_nav_agent': self.query_nav_agent,
            'get_image': self.get_image
        }
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_nav_agent",
                    "description": """
                        Send a navigation request to the navigation.

                        :param text: the intent of the user to where navigate
                        :return: a string stating if the navigation has started correctly of the object in the environment.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "the intent of the user to where navigate",
                            }
                        },
                        "required": ["text"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_image",
                    "description": """
                        Provides the camera image seen from the robot.
                        """
                }
            }
        ]

            
    def set_conversation(self) :

        self.system_instructions = f"""
        You are a service robot that has to assist any user and, eventually, navigate in an indoor 2D cartesian plane. 
        You have to give answers based on the user instructions.

        You have to understand the user intentions and based on these you have to answer properly:
            - If the user intention requires you to move into a certain spot, or place: you have to call the function 'query_nav_agent' and pass the intention of the user.
            - If the user ask to go by the tools, it means that wants to navigate by the 'tools', and you should relay this intention via 'query_nav_agent'.
            - If the user intention is referring about what do you see: you have to call the function 'get_image'. It will provide you the image from the camera of the robot.
            - When asked about what you see, you have to reason on the image provided by the tool.
            - For any other situation you can answer freely based on your BACKGROUND.
        
        BACKGROUND: 'your robot name is {self.robot_name}, and your history is explained in this web page: https://icub.iit.it/it/products/r1-robot . Other technical information are on https://www.iit.it/documents/175012/528824/Technical_Specification_R1_20200531.pdf/89e31921-f77f-96ea-b196-d89b35aba5b8?t=1623827610944 
        If you are asked anything about yourself, you have to take informations from the web page and tell to the user. Never tell the page URL.'

        """
        #More informations on the project are on this link: https://ergocub.eu/project .
        #Scientific pubblications: https://ergocub.eu/publications . People involved: https://ergocub.eu/partners . Wearable Technology: https://ergocub.eu/wearables .
        
        
        # The conversation flow contains the initial prompt and the conversation between the user and LLM
        # System role is usually the first message, that indicates the command we want to make to LLM
        # User role means the message that the user sends to the LLM
        # Assistant role is the answer of the LLM
        self.conversation_flow = [{'role': 'system', 'content': self.system_instructions}]
        
    def user_chat_callback(self, request : LlmChat.Request, response : LlmChat.Response) :
        # TODO sanity check on request
        prompt = request.text
        self.thought_flow_pub(prompt, False)
        # Add the prompt to the flow of messages
        self.conversation_flow.append({'role': 'user', 'content': prompt})
        llm_response = self.client.chat.completions.create(
            model="hsp-Vocalinteraction_gpt4o",
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
        reasoning_flag = False
        while response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"Function call: {function_name}")  
                print(f"Function arguments: {function_args}")  
                reasoning_flag = True

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
                else:
                    function_response = json.dumps({"error": "Unknown function"})
            
            llm_response = self.client.chat.completions.create(
                model="hsp-Vocalinteraction_gpt4o",
                messages=self.conversation_flow,
                max_completion_tokens=80,
                timeout=10.0
            )
            response_message = llm_response.choices[0].message
            print(f"Response Message: {response_message.content}")
            self.thought_flow_pub(response_message.content)
            self.conversation_flow.append(response_message)
        
        response.is_ok = True
        response.error_msg = ''
        
        if self.refresh_conversation and reasoning_flag:
            self.set_conversation()
            
        return response

    def query_nav_agent(self, text: str) -> Tuple:
        """
        Send a navigation request to the navigation.

        :param text: the request of the user to the navigation agent
        
        :return: a string stating if the navigation has started correctly
        """
        request = LlmChat.Request()
        request.text = text
        # Need another executor because the main thread is locked by this callback.
        # So to use the future, it's necessary to have another thread to spin the node
        self.future = self.nav_agent_client.call_async(request=request)
        self.executor.spin_until_future_complete(self.future)    # TODO add timeout and handle exception
        response : LlmChat.Response = self.future.result()

        if response.is_ok:
            return "Navigation started successfully"
        else:
            return f"Error when starting navigation: {response.error_msg}"
    
    # Returns the encoded image to pass to the LLM
    def get_image(self):
        request = CameraRGB.Request()
        self.future = self.img_client.call_async(request=request)
        self.executor.spin_until_future_complete(self.future, timeout_sec=10.0)
        response : CameraRGB.Response = self.future.result()
        if response.is_ok:
            rgb = np.frombuffer(response.rgb.data, dtype=np.uint8).reshape(response.rgb.height, response.rgb.width, 3)
            buffer = cv2.imencode('.jpg', rgb)[1]    #jpg
            encoded_img = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/jpg;base64,{encoded_img}"
        else:
            return "Image not available"
    
    def thought_flow_pub(self, text : str, is_robot = True):
        msg = String()
        if is_robot:
            msg.data = "robot: " + text
        else:
            msg.data = "user: " + text
        self.thought_pub.publish(msg)


def main(args=None):
    rclpy.init()
    node = LLMChatter("llm_chat_node")
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