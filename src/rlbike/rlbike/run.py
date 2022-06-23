import numpy as np
import random
import gym
from collections import deque
import torch
import time
import argparse
#from .files import MultiPro
#from .files.Agent import Agent
import json
import matplotlib.pyplot as plt
from math import pi

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


parser = argparse.ArgumentParser(description="")
parser.add_argument("-frames", type=int, default=50000,
                    help="The amount of training interactions with the environment, default is 1mio")
args, unknown = parser.parse_known_args()


class The_cool_bike():
    def __init__(self):
        self.max_Iq = 1000 # not sure
        self.max_q1 = 5*pi/180

    def reset(self, saved):
        self.ang = 1 # from imu
        self.state = np.array([self.ang, 0, 0], dtype=np.float32)
    
        return self.state

    def step(self, action):

        Iq_cmd = action * self.max_Iq
        Iq_cmd = Iq_cmd[0]

        q1 = 1 # from imu
        q1_dot = 1 # from imu
        q2_dot = 1 # from motor

        self.state = (q1, q1_dot, q2_dot)
        done = bool(
                q1 < -self.max_q1
                or q1 > self.max_q1
            )

        costs = 100 * q1 ** 2 + 1 * q1_dot ** 2
        
        return np.array(self.state, dtype=np.float32), -costs, done, {}


env = The_cool_bike()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

#agent = Agent(state_size=3, action_size=1, args=args, device=device)


def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def evaluate(frame, args, eval_runs=5, capture=False):
    """
    Makes an evaluation run with the current episode
    """

    reward_batch = []

    for i in range(eval_runs):

        state_action_log = np.zeros((1,4))
        #state_action_log = np.concatenate((state_action_log,[[1],[3]]),axis=1)
        #print(state_action_log)

        #state = eval_env.reset(args.saved_model)
        state = np.array([0, 0, 0], dtype=np.float32)
        rewards = 0
        rep = 0
        rep_max = args.rep_max
        if args.saved_model:
            time_duration = 15 #second
            #rep_max = time_duration/eval_env.dt
            rep_max = time_duration/1

        while True:

            # print("eval")
            # print(rend)
            rep += 1

            '''if args.render_evals:
                # print("render")
                # eval_env.render(mode="human")
                eval_env.render(i + 1)'''

            action = agent.act(np.expand_dims(state, axis=0), eval=True)
            # action = np.clip(action, action_low, action_high) <- no need, already in range (-1,+1)
            #state, reward, done, _ = eval_env.step(action[0])
            state = np.array([0, 0, 0], dtype=np.float32)
            reward = 0
            done = False


            #print(np.asmatrix(state))
            #print(np.transpose(state))
            state_action = np.append(state, action[0])
            #print(state_action)
            state_action_log = np.concatenate((state_action_log,np.asmatrix(state_action)),axis=0)
            #print(state_action_log)
            #print(rep)
            #print(len(state_action_log))

            rewards += reward
            if done or rep >= rep_max:
                rep = 0
                break

        '''if args.saved_model:
            #print(np.shape(state_action_log)[0])
            fig, axs = plt.subplots(4)
            fig.suptitle('SAC Transient Response')
            t = np.arange(0, eval_env.dt*np.shape(state_action_log)[0], eval_env.dt)
            axs[0].plot(t[1:], state_action_log[1:,0])
            axs[3].plot(t[1:], state_action_log[1:,1])
            axs[1].plot(t[1:], state_action_log[1:,2])
            axs[2].plot(t[1:], state_action_log[1:,3]*eval_env.max_torque)
            axs[0].set_ylabel('q1(rad)')
            axs[1].set_ylabel('q2 dot(rad/s)')
            axs[2].set_ylabel('torque(Nm)')
            axs[3].set_ylabel('q1 dot(rad/s)')
            axs[2].set_xlabel('time(s)')
            #axs[0].set_ylim([-0.01,0.06])
            #axs[0].set_ylim([-pi-0.5,pi+0.5])
            axs[1].set_ylim([-34,34])
            #axs[2].set_ylim([-12,12])
            plt.show()

            print("e_ss=",state_action_log[-1,0])
            print("u_ss=",state_action_log[-1,3]*eval_env.max_torque)
            print("q1_min=",min(state_action_log[1:,0]))
            print("q1_min_index=",np.argmin(state_action_log[1:,0]))
            print("OS%=",min(state_action_log[1:,0])/(eval_env.ang*pi/180))
            print("q1_a=", eval_env.ang*pi/180 * 0.9)
            print("q1_b=", eval_env.ang*pi/180 * 0.1)
            print("q1_c=", eval_env.ang*pi/180 * 0.1)
            print("q1_d=", -eval_env.ang*pi/180 * 0.1)
            min_a = 100
            min_b = 100
            min_c = 100
            min_d = 100
            t_a = 100
            t_b = 100
            t_c = 100
            t_d = 100
            for i in range(1,np.shape(state_action_log)[0]):
                tr_a = eval_env.ang*pi/180 * 0.9
                tr_b = eval_env.ang*pi/180 * 0.1
                tr_c = eval_env.ang*pi/180 * 0.1
                tr_d = -eval_env.ang*pi/180 * 0.1
                diff_a = abs(state_action_log[i,0] - tr_a)
                diff_b = abs(state_action_log[i,0] - tr_b)
                diff_c = abs(state_action_log[i,0] - tr_c)
                diff_d = abs(state_action_log[i,0] - tr_d)
                if diff_a < min_a:
                    min_a = diff_a
                    t_a = i * eval_env.dt
                if diff_b < min_b:
                    min_b = diff_b
                    t_b = i * eval_env.dt
                if diff_c < min_c:
                    min_c = diff_c
                    t_c = i * eval_env.dt
                if diff_d < min_d:
                    min_d = diff_d
                    t_d = i * eval_env.dt
            print("[min_a, t_a, min_b, t_b]=",[min_a, t_a, min_b, t_b])
            print("rising time=",t_b-t_a)
            print("[min_c, t_c, min_d, t_d]=",[min_c, t_c, min_d, t_d])
            print("settling time=",t_c,"or",t_d)'''


        reward_batch.append(rewards)
    #if capture == False and args.saved_model == False:
    #    writer.add_scalar("Reward", np.mean(reward_batch), frame)


class Node_RL(Node):

    def __init__(self):
        super().__init__('node_rl')

        self.imu_sub = self.create_subscription(
            Float64, 'list_deg', self.imu_callback, 10)
        self.imu_sub  # prevent unused variable warning

        self.motor_sub = self.create_subscription(
            Float64, 'speed_feedback', self.imu_callback, 10)
        self.motor_sub  # prevent unused variable warning

        self.frame = 1
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=100)  # last 100 scores
        self.i_episode = 1
        self.state = np.array([0, 0, 0], dtype=np.float32)
        self.score = 0
        self.frames = args.frames

        timer_period = 1 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def imu_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

    def imu_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

    def timer_callback(self):
#        self.action = agent.act(self.state)
        self.next_state = np.array([0, 0, 0], dtype=np.float32)
        self.reward = 0
        self.done = False
#        agent.step(self.state, self.action, self.reward, self.next_state, [self.done], self.frame, self.ERE)
        self.frame += 1
        self.state = self.next_state
        self.score += np.mean(self.reward)
        print("training")
        self.get_logger().info('training_info')


def main(args=args):
    rclpy.init(args=None)
    node_rl = Node_RL()
    rclpy.spin(node_rl)
    node_rl.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
