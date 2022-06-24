import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64

from .rmdx8 import *
import numpy as np
import time


class Motor(Node):

    def __init__(self):
        super().__init__('motor_bridge')

        self.motor = RmdX8()
        self.motor.cmd_send("TORQUE", np.uint16(0))
        time.sleep(1)

        self.publisher_ = self.create_publisher(Float64, 'speed_feedback', 1)
        self.subscription = self.create_subscription(
            Float64,
            'iq_cmd',
            self.listener_callback,
            10
        )
        self.subscription


    def listener_callback(self, sub_msg):
        self.get_logger().info('I heard: "%s"' % sub_msg.data)

        pub_msg = Float64()
        self.motor.cmd_send("TORQUE", np.uint16(sub_msg.data))
        pub_msg.data = float(self.motor.speed_feedback)
        self.publisher_.publish(pub_msg)
        self.get_logger().info('Publishing: "%f"' % pub_msg.data)

def main(args=None):
    rclpy.init(args=args)

    motor_m = Motor()

    rclpy.spin(motor_m)

    motor_m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
