# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import rclpy
import rclpy.duration
import rclpy.executors
from rclpy.node import Node
from ros2_vlmaps_interfaces.srv import  LlmChat
from rclpy.callback_groups import ReentrantCallbackGroup

import yarp

import numpy as np
from typing import Tuple

import yarp_callbacks

class YARPLLMWrapper(Node):
    def __init__(self, 
                 node_name : str
                 ) -> None:
        super().__init__(node_name)

        self.llm_chat_group = ReentrantCallbackGroup()  # TODO check if necessary
        self.llm_client = self.create_client(LlmChat, "/user_text", callback_group=self.llm_chat_group)

        # Wait for it to become available
        while not self.llm_client.service_is_ready():
            self.get_logger().warn(f"[{self.get_name()}] Waiting for service /llm_chat to become available")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1))

        yarp.Network.init()
        
        self.yarp_index_input_port = yarp.BufferedPortBottle()
        self.yarp_index_input_port.open("/vlmaps_llm/text:i")
        self.text_yarp_callback = yarp_callbacks.ChatCallbck(self)
        self.yarp_index_input_port.useCallback(self.text_yarp_callback)
        yarp.Network.connect("/speechTranscription_nws/text:o", "/vlmaps_llm/text:i")

    def chat(self, text: str):
        request = LlmChat.Request()
        request.text = text
        # Need another executor because the main thread is locked by this callback.
        # So to use the future, it's necessary to have another thread to spin the node
        self.future = self.llm_client.call_async(request=request)
        self.executor.spin_until_future_complete(self.future)

        response : LlmChat.Response = self.future.result()
        print(response)


def main(args=None):
    rclpy.init()
    node = YARPLLMWrapper("yarp_llm_wrapper_node")
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