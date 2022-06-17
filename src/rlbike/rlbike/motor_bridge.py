import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from rmdx8 import *
import numpy as np
import time


class Publisher(Node):

    def __init__(self):
        super().__init__('motor_bridge')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 1 # seconds
        self.motor = RmdX8()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0



    def timer_callback(self):
        msg = String()

        cmd = (self.i*100)%200
        #cmd = 0

        msg.data = 'CMD: %d' % cmd
        self.publisher_.publish(msg)

        self.motor.cmd_send("TORQUE", np.uint16(cmd))
        #self.motor.cmd_send("OFF", np.uint16(cmd))

        print(time.time())

        #self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    motor_feedback = Publisher()

    rclpy.spin(motor_feedback)

    motor_feedback.destroy_node()

    print("test")

    rclpy.shutdown()


if __name__ == '__main__':
    main()