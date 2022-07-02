import os
import time
from datetime import datetime
from math import pi
import torch
import numpy as np
import argparse
from collections import deque
#from Pendulum_v3_mirror import *
from .files.Agent import Agent
from .utils import ReplayBuffer
from .TD3 import TD3
from .PPO import PPO
#from .test import test

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


parser = argparse.ArgumentParser(description="")
parser.add_argument("-type", type=str, default="TD3", help="SAC, TD3, PPO")
parser.add_argument("-trial", type=int, default=0, help="trial")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=0.0003, help="learning rate for actor network")
parser.add_argument("-lr_c", type=float, default=0.001, help="learning rate for critic network")

#SAC arguments
parser.add_argument("-per", type=int, default=0, choices=[0, 1],
                    help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("-munchausen", type=int, default=0, choices=[0, 1],
                    help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("-dist", "--distributional", type=int, default=0, choices=[0, 1],
                    help="Using a distributional IQN Critic if set to 1, default=0")
parser.add_argument("-ere", type=int, default=0, choices=[0, 1],
                    help="Adding Emphasizing Recent Experience to the agent if set to 1, default = 0")
parser.add_argument("-n_step", type=int, default=1, help="Using n-step bootstrapping, default=1")
parser.add_argument("-d2rl", type=int, choices=[0, 1], default=0,
                    help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("--n_updates", type=int, default=1,
                    help="Update-to-Data (UTD) ratio, updates taken per step with the environment, default=1")
#parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6),
#                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--worker", type=int, default=1, help="Number of parallel worker, default = 1")
parser.add_argument("-t", "--tau", type=float, default=0.005, help="Softupdate factor tau, default is 0.005")
parser.add_argument("-layer_size", type=int, default=256,
                    help="Number of nodes per neural network layer, default is 256")
#parser.add_argument("-a", "--alpha", type=float,
#                    help="entropy alpha value, if not choosen the value is leaned by the agent")

#TD3 arguments
parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("-lr", default=1e-3, type=float, help="learning rate")

#args = parser.parse_args()
args, unknown = parser.parse_known_args()


class The_cool_bike():
    def __init__(self):
        self.max_Iq = 1000
        self.max_q1 = 3.5*pi/180
        self.max_torque = 21

    def reset(self):
        self.ang = 1 # from imu
        self.state = np.array([self.ang, 0, 0], dtype=np.float32)
    
        return np.array(self.state, dtype=np.float32)

    def step(self, action):

        #Iq_cmd = action * self.max_Iq

        #q1 = 1 # from imu
        #q1_dot = 1 # from imu
        #q2_dot = 1 # from motor
        print("inside env: ", q1, q1_dot, q2_dot)

        self.state = (q1, q1_dot, q2_dot)
        done = bool(
                q1 < -self.max_q1
                or q1 > self.max_q1
            )

        costs = 100 * q1 ** 2 + 1 * q1_dot ** 2
        
        return np.array(self.state, dtype=np.float32), -costs, done, {}


########################global variables########################

q1 = 0
q1_dot = 0
q2_dot = 0

env = The_cool_bike()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_size = 3
action_size = 1

if args.type == "SAC":
    agent = Agent(state_size=state_size, action_size=action_size, args=args, device=device)

elif args.type == "TD3":
    max_action = float(env.max_Iq)
    kwargs = {
        "state_dim": state_size,
        "action_dim": action_size,
        "max_action": max_action,
        "discount": args.gamma,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "lr": args.lr
    }
    agent = TD3(**kwargs)
    replay_buffer = ReplayBuffer(state_size, action_size)

elif args.type == "PPO":
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 10000 #args.frames / 10  # int(2.5e5)  # action_std decay frequency (in num timesteps)
    update_timestep = 2000
    K_epochs = 80
    eps_clip = 0.2
    gamma = args.gamma
    lr_actor = args.lr_a
    lr_critic = args.lr_c
    agent = PPO(state_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std)

###################### logging ######################
log_dir = f"~/runs_{args.type}/rwip{args.trial}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

checkpoint_path = log_dir + f"/rwip{args.trial}_{args.seed}.pth"

log_dir = log_dir + "/log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

log_f_name = log_dir + f"/{args.type}_log_{run_num}.csv"
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,raw_reward\n')
#####################################################

################################################################


def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def save_pth():
    if args.type == "SAC":
        torch.save(agent.actor_local.state_dict(), checkpoint_path)
    elif args.type == "TD3":
        torch.save(agent.actor.state_dict(), checkpoint_path)
    elif args.type == "PPO":
        torch.save(agent.policy_old.state_dict(), checkpoint_path)


class Node_RL(Node):

    def __init__(self):
        super().__init__('node_rl')

        self.iq_cmd_pub = self.create_publisher(Float64, 'iq_cmd', 10)

        self.imu_sub = self.create_subscription(
            Float64MultiArray, 'list_deg', self.imu_callback, 10)
        self.imu_sub  # prevent unused variable warning

        self.motor_sub = self.create_subscription(
            Float64, 'speed_feedback', self.motor_callback, 10)
        self.motor_sub  # prevent unused variable warning

        self.Iq_cmd_pub_msg = Float64()
        self.Iq_cmd_pub_msg.data = 0.0

        self.frame = 1
        self.i_episode = 1
        self.rep = 1
        self.rep_max = 500
        self.eval_every_ep = 10
        self.episode_reward = 0

        self.state = env.reset()

        timer_period = 0.005 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def imu_callback(self, msg):
        global q1
        global q1_dot
        q1 = msg.data[0]
        q1_dot = msg.data[1]
        self.get_logger().info('I heard: (%f,%f)' % (msg.data[0], msg.data[1]))

    def motor_callback(self, msg):
        global q2_dot
        q2_dot = msg.data
        self.get_logger().info('I heard: "%s"' % msg.data)

    def timer_callback(self):
        print("timestep", time.time())
        self.train()

    def pub_iq(self, action):
        self.Iq_cmd_pub_msg.data = action * env.max_Iq
        self.Iq_cmd_pub_msg.data = 25.0
        self.iq_cmd_pub.publish(self.Iq_cmd_pub_msg)
        self.get_logger().info('Publishing: "%f"' % self.Iq_cmd_pub_msg.data)

    def train(self):
        #print("timestep", time.time())

        self.rep += 1
        self.frame += 1

        if args.type == "SAC":
            action = agent.act(self.state)

            self.pub_iq(action[0])

            next_state, reward, done, _ = env.step(action[0])
            agent.step(self.state, action, reward, next_state, [done], self.frame, 0)
            self.state = next_state

        elif args.type == "TD3":
            if self.frame < args.start_timesteps:
                action = np.random.uniform(low=-env.max_Iq, high=env.max_Iq)
            else:
                action = (
                        agent.select_action(np.array(self.state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_size)
                ).clip(-max_action, max_action)

            self.pub_iq(action/env.max_Iq)

            next_state, reward, done, _ = env.step(action/env.max_Iq)
            done_bool = float(done) if self.rep < self.rep_max else 0
            replay_buffer.add(self.state, action, next_state, reward, done_bool)
            self.state = next_state
            if self.frame >= args.start_timesteps:
                agent.train(replay_buffer, args.batch_size)

        elif args.type == "PPO":
            action = agent.select_action(self.state)

            self.pub_iq(action)

            self.state, reward, done, _ = env.step(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            if self.frame % update_timestep == 0:
                agent.update()
            if self.frame % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

        self.episode_reward += reward

        if done or self.rep >= self.rep_max:
            self.rep = 0
            print(f"Episode : {self.i_episode} \t\t Timestep : {self.frame} \t\t Episode Reward : {self.episode_reward}")
            log_f.write('{},{},{}\n'.format(self.i_episode, self.frame, self.episode_reward))
            log_f.flush()
            self.i_episode += 1
            self.episode_reward = 0
            self.state = env.reset()

            save_pth()

            if self.i_episode % self.eval_every_ep == 0:
                '''eval_reward = test(env=env, agent=agent, args=args)
                print("\neval_reward", eval_reward)
                if eval_reward > -10:
                    #break
                    pass'''
                pass


def main(args=args):
    rclpy.init(args=None)
    node_rl = Node_RL()
    rclpy.spin(node_rl)
    node_rl.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":

    t0 = time.time()

    main()

    t1 = time.time()
    timer(t0, t1)
    env.close()
    log_f.close()
