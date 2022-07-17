import time
from rmdx8 import *
import matplotlib.pyplot as plt
from OpenIMU_SPI import *
import RPi.GPIO as GPIO

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

my_motor = RmdX8()
openimu_spi = SpiOpenIMU(target_module="300ZI", fw='26.0.7', cs_pin = 19, interrupt_pin = 26, drdy_status=False)

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
