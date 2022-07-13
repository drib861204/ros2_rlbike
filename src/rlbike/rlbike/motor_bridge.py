import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64

from .rmdx8 import *
import numpy as np
import time


###global variables###
global_iq_cmd = 0
######################

class Motor(Node):

    def __init__(self):
        super().__init__('motor_bridge')

        self.motor = RmdX8()
        self.motor.cmd_send("TORQUE", np.uint16(0))

        self.publisher_ = self.create_publisher(Float64, 'speed_feedback', 10)
        self.speed_fb_msg = Float64()
        self.subscription = self.create_subscription(
            Float64,
            'iq_cmd',
            self.listener_callback,
            10
        )
        self.subscription

        timer_period = 0.01 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def listener_callback(self, iq_cmd_msg):
        global global_iq_cmd

        global_iq_cmd = iq_cmd_msg.data
        self.get_logger().info('I heard iq cmd: "%s"' % global_iq_cmd)

    def timer_callback(self):
        print(time.time())
        global global_iq_cmd
        print("iq_cmd in timer callback", global_iq_cmd)
        self.motor.cmd_send("TORQUE", np.uint16(global_iq_cmd))

        self.speed_fb_msg.data = float(self.motor.speed_feedback)
        self.publisher_.publish(self.speed_fb_msg)
        self.get_logger().info('speed feedback: "%f"' % self.speed_fb_msg.data)

def main(args=None):
    rclpy.init(args=args)
    motor_m = Motor()
    rclpy.spin(motor_m)
    motor_m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
