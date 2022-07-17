import matplotlib
matplotlib.use('Agg')
import os
import time
import torch
import numpy as np
from math import pi
import argparse
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray







Iq_feedback = np.int(0)
q2_dot = np.int(0)
Iq_cmd = 0
q1 = 0
q1_dot = 0

my_motor.cmd_send("TORQUE", np.uint16(0))
time.sleep(2)

for _ in range(1000):
    #------------------------------------------------#
    #OpenIMU get q1 & q1 dot
        list_rate, list_acc, list_deg = openimu_spi.burst_read(first_register=0x3D,subregister_num=11)
        str_burst = "time:{0:>10f};  gyro:{1:>25s};  accel:{2:>25s};  deg:{2:>25s} \n".format(
            time.clock(), ", ".join([str(x) for x in list_rate]), ", ".join([str(x) for x in list_acc]), ", ".join([str(x) for x in list_deg])
            )
        print(str_burst)
        q1 = list_deg[1]
        q1_dot = list_rate[1]
        
    #------------------------------------------------#
    #PID control
    kp = 0.001
    Iq_cmd = -kp * q1


    #------------------------------------------------#
    #RMD-X8 get q2 dot & set current command
    my_motor.cmd_send("TORQUE", np.uint16(Iq_cmd))
    time.sleep(0.005)
    q2_dot = my_motor.speed_feedback
