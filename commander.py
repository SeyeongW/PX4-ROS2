import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import sys

# ==========================================
# [Target Commander Node]
# - Only sends Target ID to the drone
# ==========================================
class CommanderNode(Node):
    def __init__(self):
        super().__init__('commander_node')
        
        # Publisher to send ID
        self.publisher = self.create_publisher(Int32, '/perception/set_target_id', 10)
        print("------------------------------------------")
        print("   Drone Target Commander Initialized")
        print("------------------------------------------")
        print(" Type target ID and press Enter.")
        print(" Example: 5 -> tracks ID 5")
        print(" Type 'q' to quit.")
        print("------------------------------------------")

    def send_id(self, target_id):
        msg = Int32()
        msg.data = int(target_id)
        self.publisher.publish(msg)
        print(f" [Sent] Command sent to track ID: {target_id}")

def main():
    rclpy.init()
    node = CommanderNode()

    try:
        while rclpy.ok():
            # Get user input
            user_input = input("\nTarget ID >> ")
            
            if user_input.lower() == 'q':
                break
            
            if user_input.strip() == "":
                continue

            try:
                # Check if input is a valid integer
                target_id = int(user_input)
                node.send_id(target_id)
            except ValueError:
                print(" [Error] Please enter a valid number.")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
