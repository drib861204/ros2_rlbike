import os
import time
import torch
import numpy as np
from math import pi
import argparse
import matplotlib.pyplot as plt
from .files.Agent import Agent
from .TD3 import TD3
from .PPO import PPO


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


parser = argparse.ArgumentParser(description="")
parser.add_argument("-type", type=str, default="SAC", help="SAC, TD3, PPO")
parser.add_argument("-trial", type=int, default=101, help="trial")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=0.0003, help="learning rate for actor network")
parser.add_argument("-lr_c", type=float, default=0.001, help="learning rate for critic network")

# SAC parameters
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
parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel worker, default = 1")
parser.add_argument("-t", "--tau", type=float, default=0.005, help="Softupdate factor tau, default is 0.005")
parser.add_argument("-layer_size", type=int, default=256,
                    help="Number of nodes per neural network layer, default is 256")
#parser.add_argument("-a", "--alpha", type=float,
#                    help="entropy alpha value, if not choosen the value is leaned by the agent")

# TD3 parameters
parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("-lr", default=3e-4, type=float, help="learning rate")

#args = parser.parse_args()
args, unknown = parser.parse_known_args()


class The_cool_bike():
    def __init__(self):
        self.max_q1 = 3.5*pi/180 # rad
        self.max_q1dot = 0.3 # rad/sec
        self.max_Iq = 1100
        self.wheel_max_speed = 28 # rad/sec

        self.reset_ang = 1*pi/180 # rad

        self.Iq_cmd = 0
        self.last_Iq_cmd = 0

        self.q1dot_log = []

    def reset(self):
        global q1
        global q1_dot
        global q2_dot
        reset_q1 = q1
        reset_q1_dot = q1_dot
        reset_q2_dot = q2_dot

        self.state = np.array([reset_q1, reset_q1_dot, reset_q2_dot], dtype=np.float32)
        '''if reset_q1 >= 0:
            self.agent_state = np.array([reset_q1, reset_q1_dot, reset_q2_dot], dtype=np.float32)
        else:
            self.agent_state = np.array([-reset_q1, -reset_q1_dot, -reset_q2_dot], dtype=np.float32)'''

        self.agent_state = self.norm_agent_state(self.state)

        self.last_Iq_cmd = 0

        return np.array(self.agent_state, dtype=np.float32)

    def step(self, action):
        global q1
        global q1_dot
        global q2_dot
        env_q1 = q1
        env_q1_dot = q1_dot
        env_q2_dot = q2_dot

        self.q1dot_log.append(env_q1_dot)
        print("q2 dot mean: ", np.mean(self.q1dot_log))

        self.Iq_cmd = action * self.max_Iq
        self.state = (env_q1, env_q1_dot, env_q2_dot)

        '''if env_q1 >= 0:
            self.Iq_cmd = action * self.max_Iq
            self.agent_state = (env_q1, env_q1_dot, env_q2_dot)
        else:
            self.Iq_cmd = -action * self.max_Iq
            self.agent_state = (-env_q1, -env_q1_dot, -env_q2_dot)'''
        self.agent_state = self.norm_agent_state(self.state)

        #print("inside env: ", q1, q1_dot, q2_dot)

        done = bool(
            env_q1 < -self.max_q1
            or env_q1 > self.max_q1
            or env_q1_dot < -self.max_q1dot
            or env_q1_dot > self.max_q1dot
        )

        # for reward calculating
        torque = self.Iq_cmd * 21/self.max_Iq
        last_torque = self.last_Iq_cmd * 21/self.max_Iq

        costs = 100 * env_q1 ** 2 + 1 * env_q1_dot ** 2 + 0.0001 * (last_torque - torque) ** 2
        if done:
            costs += 100

        self.last_Iq_cmd = self.Iq_cmd

        return np.array(self.agent_state, dtype=np.float32), -costs, done, {}

    def norm_agent_state(self, state):
        state = ((state[0] - self.max_q1) / (2 * self.max_q1),
                 (state[1] - self.max_q1dot) / (2 * self.max_q1dot),
                 (state[2] - self.wheel_max_speed) / (2 * self.wheel_max_speed)
        )
        return state


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

log_dir = f"/home/ptlab/ros2_rlbike/runs_{args.type}/rwip{args.trial}"
checkpoint_path = log_dir + f"/rwip{args.trial}_0.pth"
log_dir = log_dir + "/fig"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if args.type == "SAC":
    agent = Agent(state_size=state_size, action_size=action_size, args=args, device=device)
    agent.actor_local.load_state_dict(torch.load(checkpoint_path, map_location=device))

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
        "lr_a": args.lr_a,
        "lr_c": args.lr_c
    }
    agent = TD3(**kwargs)
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

elif args.type == "PPO":
    action_std = 0.001  # starting std for action distribution (Multivariate Normal)
    K_epochs = 80
    eps_clip = 0.2
    gamma = args.gamma
    lr_actor = args.lr_a
    lr_critic = args.lr_c
    agent = PPO(state_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, True, action_std)
    agent.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

################################################################


def transient_response(env, state_action_log, type, seconds):
    fig, axs = plt.subplots(4)
    fig.suptitle(f'{type} Transient Response')
    t = np.linspace(0, seconds, state_action_log.shape[0])
    axs[0].plot(t[1:], state_action_log[1:,0])
    axs[1].plot(t[1:], state_action_log[1:,1])
    axs[2].plot(t[1:], state_action_log[1:,2])
    axs[3].plot(t[1:], state_action_log[1:,3]*21)
    axs[0].set_ylabel('q1(deg)')
    axs[1].set_ylabel('q1 dot(deg/s)')
    axs[2].set_ylabel('q2 dot(deg/s)')
    axs[3].set_ylabel('torque(Nm)')
    axs[3].set_xlabel('time(s)')
    '''axs[0].set_ylim([-0.065,0.065])
    axs[1].set_ylim([-0.15,0.15])
    axs[2].set_ylim([-34,34])
    axs[3].set_ylim([-24,24])'''
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[2].get_xaxis().set_visible(False)

    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    plt.savefig(log_dir + f"/response{run_num}")
    #plt.show()

    '''print("e_ss=",state_action_log[-1,0])
    print("u_ss=",state_action_log[-1,3]*env.max_torque)
    print("q1_min=",min(state_action_log[1:,0]))
    print("q1_min_index=",np.argmin(state_action_log[1:,0]))
    print("OS%=",min(state_action_log[1:,0])/(env.ang*pi/180))
    print("q1_a=", env.ang*pi/180 * 0.9)
    print("q1_b=", env.ang*pi/180 * 0.1)
    print("q1_c=", env.ang*pi/180 * 0.1)
    print("q1_d=", -env.ang*pi/180 * 0.1)
    min_a = 100
    min_b = 100
    min_c = 100
    min_d = 100
    t_a = 100
    t_b = 100
    t_c = 100
    t_d = 100
    for i in range(1,np.shape(state_action_log)[0]):
        tr_a = env.ang*pi/180 * 0.9
        tr_b = env.ang*pi/180 * 0.1
        tr_c = env.ang*pi/180 * 0.1
        tr_d = -env.ang*pi/180 * 0.1
        diff_a = abs(state_action_log[i,0] - tr_a)
        diff_b = abs(state_action_log[i,0] - tr_b)
        diff_c = abs(state_action_log[i,0] - tr_c)
        diff_d = abs(state_action_log[i,0] - tr_d)
        if diff_a < min_a:
            min_a = diff_a
            t_a = i * env.dt
        if diff_b < min_b:
            min_b = diff_b
            t_b = i * env.dt
        if diff_c < min_c:
            min_c = diff_c
            t_c = i * env.dt
        if diff_d < min_d:
            min_d = diff_d
            t_d = i * env.dt
    print("[min_a, t_a, min_b, t_b]=",[min_a, t_a, min_b, t_b])
    print("rising time=",t_b-t_a)
    print("[min_c, t_c, min_d, t_d]=",[min_c, t_c, min_d, t_d])
    print("settling time=",t_c,"or",t_d)'''


class Node_RL(Node):

    def __init__(self):
        super().__init__('node_rl')

        self.iq_cmd_pub = self.create_publisher(Float64, 'iq_cmd', 10)
        self.Iq_cmd_pub_msg = Float64()
        self.Iq_cmd_pub_msg.data = 0.0

        self.imu_sub = self.create_subscription(
            Float64MultiArray, 'imu_data', self.imu_callback, 10)
        self.imu_sub  # prevent unused variable warning

        self.motor_sub = self.create_subscription(
            Float64, 'speed_feedback', self.motor_callback, 10)
        self.motor_sub  # prevent unused variable warning

        self.frame = 0
        self.i_episode = 1
        self.rep = 0
        self.rep_max = 500
        self.plot_response_freq = 10
        self.episode_reward = 0
        self.state_action_log = np.zeros((1, 4))

        self.in_reset_range = False
        self.testing = False

        #self.state = env.reset()
        self.action = 0.0
        self.pub_iq(0.0)

        timer_period = 0.05 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def imu_callback(self, msg):
        global q1
        global q1_dot
        q1 = msg.data[0]
        q1_dot = msg.data[1]
        #self.get_logger().info('IMU data: (%f,%f)' % (msg.data[0], msg.data[1]))

        if abs(q1) < env.reset_ang:
            self.in_reset_range = True
        else:
            self.in_reset_range = False

    def motor_callback(self, msg):
        global q2_dot
        q2_dot = msg.data
        #self.get_logger().info('motor speed feedback: "%s"' % msg.data)

    def timer_callback(self):
        #print("timestep", time.time())
        if not self.testing:
            if not self.in_reset_range:
                print(f"Waiting for reset: |q1| < {env.reset_ang} ...")
            else:
                print("Reset ok!")
                self.state = env.reset()
                self.pub_iq(0.0)
                self.testing = True
                self.t0 = time.time()
        else:
            self.test()

    def pub_iq(self, action):
        self.action = action

        #print(action)
        self.Iq_cmd_pub_msg.data = action * env.max_Iq
        self.Iq_cmd_pub_msg.data = 0.0
        self.iq_cmd_pub.publish(self.Iq_cmd_pub_msg)
        self.get_logger().info('Publishing Iq cmd: "%f"' % self.Iq_cmd_pub_msg.data)

    def test(self):

        self.rep += 1
        self.frame += 1

        self.state, reward, done, _ = env.step(self.action)

        state_for_render = env.state
        state_action = np.append(state_for_render, self.action)
        self.state_action_log = np.concatenate((self.state_action_log, np.asmatrix(state_action)), axis=0)
        self.episode_reward += reward

        if done or self.rep >= self.rep_max:
            self.testing = False

            self.rep = 0
            print(f"Episode : {self.i_episode} \t\t Timestep : {self.frame} \t\t Episode Reward : {self.episode_reward}")
            self.i_episode += 1
            self.episode_reward = 0
            self.pub_iq(0.0)

            print("Wait for 5 seconds to reset")
            time.sleep(1)
            print("4...")
            time.sleep(1)
            print("3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)

        else:
            if self.rep % self.plot_response_freq == 0:
                t1 = time.time()
                hours, seconds = divmod((t1-self.t0), 3600)
                transient_response(env, self.state_action_log, args.type, seconds)
                #print("cumulative reward:", self.episode_reward)

            if args.type == "SAC":
                action_cmd = agent.act(np.expand_dims(self.state, axis=0), eval=True)
                action_cmd = action_cmd[0][0]
            elif args.type == "TD3":
                action_cmd = agent.select_action(np.array(self.state))
                action_cmd /= env.max_Iq
            elif args.type == "PPO":
                action_cmd = agent.select_action(self.state, test=True)

            self.pub_iq(action_cmd)


def main(args=args):
    rclpy.init(args=None)
    node_rl = Node_RL()
    rclpy.spin(node_rl)
    node_rl.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":

    main()

    env.close()
