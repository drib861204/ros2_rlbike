import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64

#from .rmdx8 import *
import numpy as np
import time


class Motor(Node):

    def __init__(self):
        super().__init__('motor_bridge')
        self.publisher_ = self.create_publisher(Float64, 'speed_feedback', 10)
        self.subscription = self.create_subscription(
            Float64,
            'iq_cmd',
            self.listener_callback,
            10
        )
        self.subscription
        timer_period = 1 # seconds

        #self.motor = RmdX8()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


    def timer_callback(self):
        msg = Float64()

        Iq_cmd = 20 # for test

        '''self.motor.cmd_send("TORQUE", np.uint16(Iq_cmd))
        time.sleep(0.005)
        q2_dot = self.motor.speed_feedback


        msg.data = q2_dot'''
        msg.data = 55.0
        #self.publisher_.publish(msg)

        #print(time.time())

        self.get_logger().info('Publishing: "%f"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    motor_m = Motor()
    
    while(1):
        print("test1")

    rclpy.spin(motor_m)

    print("test2")

    motor_m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
