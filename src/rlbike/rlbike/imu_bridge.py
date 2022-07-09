from math import pi
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

from .OpenIMU_SPI import *


class IMU(Node):

    def __init__(self):
        super().__init__('imu_bridge')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'imu_data', 10)

        nRST_PIN = 21
        time.sleep(0.1)
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(nRST_PIN, GPIO.OUT)
        time.sleep(0.1)
        GPIO.output(nRST_PIN, 0)
        time.sleep(5)
        GPIO.output(nRST_PIN, 1)
        print("Reset Ready")
        time.sleep(0.1)

        self.q1 = 0
        self.q1_dot = 0
        self.bias_q1 = 0.5
        self.bias_q1dot = 0.04

        self.openimu_spi = SpiOpenIMU(target_module="300ZI", fw='26.0.7', cs_pin = 19, interrupt_pin = 26, drdy_status=False)
        
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()

        list_rate, list_acc, list_deg = self.openimu_spi.burst_read(first_register=0x3D,subregister_num=11)
        #str_burst = "time:{0:>10f};  gyro:{1:>25s};  accel:{2:>25s};  deg:{3:>25s} \n".format(
        #    time.clock(), ", ".join([str(x) for x in list_rate]), ", ".join([str(x) for x in list_acc]), ", ".join([str(x) for x in list_deg])
        #    )
        #print(str_burst)
        #q1 = list_deg[1]
        #q1_dot = list_rate[1]
        #print(list_deg)

        if list_deg[0] >= 0:
            self.q1 = list_deg[0] - 180 - self.bias_q1
        else:
            self.q1 = list_deg[0] + 180 - self.bias_q1

        self.q1_dot = list_rate[0] - self.bias_q1dot

        msg.data = [self.q1*pi/180, self.q1_dot*pi/180]
        #msg.data = [self.q1*1000, self.q1_dot*1000]
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing imu data (%f,%f)' % (msg.data[0], msg.data[1]))

def main(args=None):
    rclpy.init(args=args)
    imu_m = IMU()
    rclpy.spin(imu_m)
    imu_m.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
