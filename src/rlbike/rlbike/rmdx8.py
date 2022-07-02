import numpy as np
import can
import time
from math import pi

class RmdX8:
    def __init__(self):
        self.RMD_X8_SPEED_LIMITED = 514
        self.SET_TORQUE_CMD = "TORQUE"
        self.MOTOR_OFF = "OFF"
        self.STATUS = "STATUS"
        
        self.speed_feedback = np.int16(0)

        self.bus = can.Bus(receive_own_messages=True)

    def cmd_send(self, cmd, Iq): # data = np.uint16(value)
        if cmd == self.SET_TORQUE_CMD:
            print("Set torque for motor run")
            data = [0xA1, 0x00, 0x00, 0x00, Iq & 0xFF, Iq >> 8, 0x00, 0x00]
            self.send_one(data)
            #print(data)

        if cmd == self.MOTOR_OFF:
            print("Turn off motor")
            data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            self.send_one(data)

        if cmd == self.STATUS:
            print("Read motor status")
            data = [0x9C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            self.send_one(data)

    def send_one(self, data=None):
        if data is None:
            data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        with can.ThreadSafeBus() as bus:
            msg = can.Message(
                arbitration_id=0x141, data=data, is_extended_id=False
            )

            try:
                bus.send(msg)
                #time.sleep(0.01)
                msg_recv = bus.recv(timeout=0.005)
                #print(f"Message sent on {bus.channel_info}")
                #print(f"Message: {msg_recv}")
                #print(f"data: {msg_recv.data}")
                #print(f"data[3]: {hex(msg_recv.data[3])}")
                #print(f"data[2]: {hex(msg_recv.data[2])}")
                #self.Iq_feedback = np.int16((msg_recv.data[3]<<8)+(msg_recv.data[2]))
                #print(f"Iq_feedback: {self.Iq_feedback}")

                #print(f"data[5]: {hex(msg_recv.data[5])}")
                #print(f"data[4]: {hex(msg_recv.data[4])}")

                if msg_recv == None:
                    pass
                    #self.speed_feedback = self.speed_feedback
                else:
                    self.speed_feedback = np.int16((msg_recv.data[5]<<8)+(msg_recv.data[4]))
                print(f"speed_feedback: {self.speed_feedback}")

                #self.position_feedback = np.int16((msg_recv.data[7]<<8)+(msg_recv.data[6]))
                #print(f"speed_feedback: {self.position_feedback}")

            except can.CanError:
                print("Message NOT sent")
