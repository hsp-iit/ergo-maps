# SPDX-FileCopyrightText: 2024 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause
# Author: Simone Micheletti

import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Using String messages for chat
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

# Initialize Flask
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class RosChatBackend(Node):
    def __init__(self):
        super().__init__("chat_backend_subscriber")
        self.subscription = self.create_subscription(
            String, "llm_reasoning", self.callback, 10
        )

    def callback(self, msg : String):
        print(f"Received message: {msg.data}")
        socketio.emit("chat_message", {"message": msg.data})  # Send to frontend

def run_ros2_node():
    rclpy.init()
    node = RosChatBackend()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# Update the html page
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Start ROS 2 node in a separate process
    ros2_thread = threading.Thread(target=run_ros2_node, daemon=True)
    ros2_thread.start()

    # Start Flask app with WebSocket support
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)