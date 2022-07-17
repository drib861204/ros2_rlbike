import matplotlib
matplotlib.use('Agg')
import os
import time
from datetime import datetime
from math import pi
import torch
import numpy as np
import argparse
from collections import deque
import dill
from .files.Agent import Agent
from .utils import ReplayBuffer
from .TD3 import TD3
from .PPO import PPO
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


parser = argparse.ArgumentParser(description="")
parser.add_argument("-type", type=str, default="SAC", help="SAC, TD3, PPO")
parser.add_argument("-trial", type=int, default=0, help="trial")
parser.add_argument("-cont_trial", type=int, default=0, help="continue training trial")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=0.0003, help="learning rate for actor network")
parser.add_argument("-lr_c", type=float, default=0.001, help="learning rate for critic network")
parser.add_argument("-torque_delay", type=int, default=2, help="consider torque delay. 1: state + last_torque, 2: + last and current torque, 3: original state")
parser.add_argument("-two_state", type=int, default=0, help="only q1 and q2 dot")

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
        self.max_q1 = 3.5*pi/180 # rad
        self.max_q1dot = 0.3 # rad/sec
        self.max_Iq = 1100*11/21 #1100
        self.max_torque = 11 #21
        self.wheel_max_speed = 28 # rad/sec

        self.reset_ang = 2*pi/180 # rad

        self.Iq_cmd = 0
        self.last_Iq_cmd = 0

    def reset(self):
        global q1
        global q1_dot
        global q2_dot
        reset_q1 = np.clip(q1, -self.max_q1, self.max_q1)
        reset_q1_dot = q1_dot
        reset_q2_dot = np.clip(q2_dot, -self.wheel_max_speed, self.wheel_max_speed)

        self.last_Iq_cmd = 0

        self.state = np.array([reset_q1, reset_q1_dot, reset_q2_dot], dtype=np.float32)

        self.agent_state = self.norm_agent_state(self.state)

        if args.torque_delay == 1:
            self.state_delay = np.append(np.array(self.agent_state, dtype=np.float32), [-0.5])
            return self.state_delay
        elif args.torque_delay == 2:
            self.state_delay = np.append(np.array(self.agent_state, dtype=np.float32), [-0.5, -0.5])
            return self.state_delay
        elif args.torque_delay == 3:
            self.last2_Iq_cmd = 0
            self.state_delay = np.append(np.array(self.agent_state, dtype=np.float32), [-0.5, -0.5, -0.5])
            return self.state_delay
        else:
            return np.array(self.agent_state, dtype=np.float32)

    def step(self, action):
        global q1
        global q1_dot
        global q2_dot
        env_q1 = np.clip(q1, -self.max_q1, self.max_q1)
        env_q1_dot = q1_dot
        env_q2_dot = np.clip(q2_dot, -self.wheel_max_speed, self.wheel_max_speed)

        self.Iq_cmd = action * self.max_Iq
        self.state = (env_q1, env_q1_dot, env_q2_dot)
        '''if env_q1 >= 0:
            self.Iq_cmd = action * self.max_Iq
            self.agent_state = (env_q1, env_q1_dot, env_q2_dot)
        else:
            self.Iq_cmd = -action * self.max_Iq
            self.agent_state = (-env_q1, -env_q1_dot, -env_q2_dot)'''
        self.agent_state = self.norm_agent_state(self.state)

        #print("inside env: ", q1, q1_dot, q2_dot, Iq_cmd)
        #print("inside env type: ", type(q1), type(q1_dot), type(q2_dot), type(Iq_cmd))
        #self.state = (env_q1, env_q1_dot, env_q2_dot)

        done = bool(
            env_q1 <= -self.max_q1
            or env_q1 >= self.max_q1
        )
            #or env_q1_dot < -self.max_q1dot
            #or env_q1_dot > self.max_q1dot

        costs = 100 * env_q1 ** 2 + 1 * env_q1_dot ** 2 + 0.001 * env_q2_dot ** 2
        if done:
            costs += 100

        if args.torque_delay == 1:
            last_Iq_norm = (self.last_Iq_cmd - self.max_Iq) / (2 * self.max_Iq)
            self.state_delay = np.append(np.array(self.agent_state, dtype=np.float32), [last_Iq_norm])
            #print(np.array(self.agent_state, dtype=np.float32))
            self.last_Iq_cmd = self.Iq_cmd
            return self.state_delay, -costs, done, {}

        elif args.torque_delay == 2:
            last_Iq_norm = (self.last_Iq_cmd - self.max_Iq) / (2 * self.max_Iq)
            Iq_norm = (self.Iq_cmd - self.max_Iq) / (2 * self.max_Iq)
            self.state_delay = np.append(np.array(self.agent_state, dtype=np.float32), [last_Iq_norm, Iq_norm])
            self.last_Iq_cmd = self.Iq_cmd
            return self.state_delay, -costs, done, {}

        elif args.torque_delay == 3:
            last2_Iq_norm = (self.last2_Iq_cmd - self.max_Iq) / (2 * self.max_Iq)
            last_Iq_norm = (self.last_Iq_cmd - self.max_Iq) / (2 * self.max_Iq)
            Iq_norm = (self.Iq_cmd - self.max_Iq) / (2 * self.max_Iq)
            self.state_delay = np.append(np.array(self.agent_state, dtype=np.float32), [last2_Iq_norm, last_Iq_norm, Iq_norm])
            self.last_Iq_cmd = self.Iq_cmd
            self.last2_Iq_cmd = self.last_Iq_cmd
            return self.state_delay, -costs, done, {}

        else:
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

if args.torque_delay == 1:
    state_size = 4
elif args.torque_delay == 2:
    state_size = 5
elif args.torque_delay == 3:
    state_size = 6
else:
    state_size = 3
action_size = 1

if args.type == "SAC":
    agent = Agent(state_size=state_size, action_size=action_size, args=args, device=device)

    if args.cont_trial:
        model_pth = f"{os.path.abspath(os.getcwd())}/runs_SAC/rwip{args.cont_trial}/rwip{args.cont_trial}_{args.seed}"
        agent.actor_local.load_state_dict(torch.load(model_pth+"_actor", map_location=torch.device('cpu')))
        agent.actor_optimizer.load_state_dict(torch.load(model_pth+"_actor_optimizer", map_location=torch.device('cpu')))
        agent.critic1.load_state_dict(torch.load(model_pth+"_critic1", map_location=torch.device('cpu')))
        agent.critic1_target.load_state_dict(agent.critic1.state_dict())
        agent.critic1_optimizer.load_state_dict(torch.load(model_pth+"_critic1_optimizer", map_location=torch.device('cpu')))
        agent.critic2.load_state_dict(torch.load(model_pth+"_critic2", map_location=torch.device('cpu')))
        agent.critic2_target.load_state_dict(agent.critic2.state_dict())
        agent.critic2_optimizer.load_state_dict(torch.load(model_pth+"_critic2_optimizer", map_location=torch.device('cpu')))
        agent.log_alpha = torch.load(model_pth+"_log_alpha", map_location=torch.device('cpu'))
        agent.alpha_optimizer.load_state_dict(torch.load(model_pth+"_alpha_optimizer", map_location=torch.device('cpu')))
        agent.memory = dill.load(open(model_pth+"_memory", "rb"))

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
log_dir = f"{os.path.abspath(os.getcwd())}/runs_{args.type}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_files = os.listdir(log_dir)
files_num = len(current_files)

log_dir = log_dir + f"/rwip{args.trial}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#checkpoint_path = log_dir + f"/rwip{files_num}_{run_num}.pth"
checkpoint_path = log_dir + f"/rwip{args.trial}_{run_num}"

fig_dir = log_dir + "/fig_training"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

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
        #torch.save(agent.actor_local.state_dict(), checkpoint_path)
        torch.save(agent.actor_local.state_dict(), checkpoint_path+"_actor")
        torch.save(agent.actor_optimizer.state_dict(), checkpoint_path+"_actor_optimizer")

        torch.save(agent.critic1.state_dict(), checkpoint_path+"_critic1")
        torch.save(agent.critic1_optimizer.state_dict(), checkpoint_path+"_critic1_optimizer")

        torch.save(agent.critic2.state_dict(), checkpoint_path+"_critic2")
        torch.save(agent.critic2_optimizer.state_dict(), checkpoint_path+"_critic2_optimizer")

        torch.save(agent.log_alpha, checkpoint_path+"_log_alpha")
        torch.save(agent.alpha_optimizer.state_dict(), checkpoint_path+"_alpha_optimizer")

        dill.dump(agent.memory, file = open(checkpoint_path+"_memory", "wb"))

    elif args.type == "TD3":
        torch.save(agent.actor.state_dict(), checkpoint_path)
    elif args.type == "PPO":
        torch.save(agent.policy_old.state_dict(), checkpoint_path)


def transient_response(env, state_action_log, type, seconds, fig_file_name):
    fig, axs = plt.subplots(4)
    fig.suptitle(f'{type} Transient Response')
    t = np.linspace(0, seconds, state_action_log.shape[0])
    axs[0].plot(t[1:], state_action_log[1:,0])
    axs[1].plot(t[1:], state_action_log[1:,1])
    axs[2].plot(t[1:], state_action_log[1:,2])
    axs[3].plot(t[1:], state_action_log[1:,3]*env.max_torque)
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')
    axs[2].grid(axis='y')
    axs[3].grid(axis='y')
    axs[0].set_ylabel('q1(rad)')
    axs[1].set_ylabel('q1 dot(rad/s)')
    axs[2].set_ylabel('q2 dot(rad/s)')
    axs[3].set_ylabel('torque(Nm)')
    axs[3].set_xlabel('time(s)')
    '''axs[0].set_ylim([-0.065,0.065])
    axs[1].set_ylim([-0.15,0.15])
    axs[2].set_ylim([-34,34])
    axs[3].set_ylim([-24,24])'''
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[2].get_xaxis().set_visible(False)

    plt.savefig(fig_file_name)
    plt.close()
    #plt.show()


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
        self.rep_max = 30 #500
        self.episode_reward = 0
        self.state_action_log = np.zeros((1, 4))

        self.in_reset_range = False
        self.training = False

        self.action = 0.0
        #self.state = env.reset()
        #self.pub_iq(0.0)

        self.fig_file_name = "."
        self.t0 = time.time()

        timer_period = 0.05 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def imu_callback(self, msg):
        global q1
        global q1_dot
        q1 = msg.data[0]
        q1_dot = msg.data[1]

        if abs(msg.data[0]) < env.reset_ang and abs(msg.data[1]) < 0.05:
            self.in_reset_range = True
        else:
            self.in_reset_range = False

        #self.get_logger().info('IMU data: (%f,%f)' % (msg.data[0], msg.data[1]))

    def motor_callback(self, msg):
        global q2_dot
        q2_dot = msg.data
        #self.get_logger().info('motor speed feedback: "%s"' % msg.data)

    def alg_action(self):
        action_cmd = 0
        if args.type == "SAC":
            #agent.step(self.state, self.action, reward, next_state, [done], self.frame, 0)
            #self.state = next_state

            action_cmd = agent.act(self.state)

            action_cmd = action_cmd[0]

        self.pub_iq(action_cmd)

    def timer_callback(self):
        #print("timestep", time.time())
        if not self.training:
            if not self.in_reset_range:
                pass
                #print(f"Waiting for reset: |q1| < {env.reset_ang} ...")
            else:
                print("Reset ok!")

                current_num_files = next(os.walk(fig_dir))[2]
                run_num = len(current_num_files)
                self.fig_file_name = fig_dir + f"/response{run_num}"

                self.state_action_log = np.zeros((1, 4))

                self.state = env.reset()

                self.alg_action()

                #self.pub_iq(0.0)
                self.training = True
                self.t0 = time.time()
        else:
            self.train()

    def pub_iq(self, action):
        self.action = action

        self.Iq_cmd_pub_msg.data = self.action * env.max_Iq
        
        if self.frame % 2 == 0:
            self.Iq_cmd_pub_msg.data = 200.0
        else:
            self.Iq_cmd_pub_msg.data = -200.0
            
        self.iq_cmd_pub.publish(self.Iq_cmd_pub_msg)
        #self.get_logger().info('Publishing Iq cmd: "%f"' % self.Iq_cmd_pub_msg.data)

    def train(self):

        self.rep += 1
        self.frame += 1

        next_state, reward, done, _ = env.step(self.action)
        self.episode_reward += reward

        if args.type == "SAC":
            agent.step(self.state, self.action, reward, next_state, [done], self.frame, 0)
            self.state = next_state

        elif args.type == "TD3":
            done_bool = float(done) if self.rep < self.rep_max else 0
            replay_buffer.add(self.state, self.action, next_state, reward, done_bool)
            self.state = next_state
            if self.frame >= args.start_timesteps:
                agent.train(replay_buffer, args.batch_size)

        elif args.type == "PPO":
            self.state = next_state
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            if self.frame % update_timestep == 0:
                agent.update()
            if self.frame % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

        state_for_render = env.state
        state_action = np.append(state_for_render, self.action)
        self.state_action_log = np.concatenate((self.state_action_log, np.asmatrix(state_action)), axis=0)

        if done or self.rep >= self.rep_max:
            self.training = False

            self.rep = 0
            print(f"Episode : {self.i_episode} \t\t Timestep : {self.frame} \t\t Episode Reward : {self.episode_reward}")
            log_f.write('{},{},{}\n'.format(self.i_episode, self.frame, self.episode_reward))
            log_f.flush()
            self.i_episode += 1
            self.episode_reward = 0
            #self.state = env.reset()
            self.pub_iq(0.0)

            t1 = time.time()
            hours, seconds = divmod((t1-self.t0), 3600)
            transient_response(env, self.state_action_log, args.type, seconds, self.fig_file_name)

            if self.frame % 1000 < 20:
                print("save model...")
                save_pth()

            print("Wait for 1 second to reset")
            time.sleep(1)

        else:
            if args.type == "SAC":
                action_cmd = agent.act(self.state)
                action_cmd = action_cmd[0]

            elif args.type == "TD3":
                if self.frame < args.start_timesteps:
                    action_cmd = np.random.uniform(low=-env.max_Iq, high=env.max_Iq)
                else:
                    action_cmd = (
                            agent.select_action(np.array(self.state))
                            + np.random.normal(0, max_action * args.expl_noise, size=action_size)
                    ).clip(-max_action, max_action)

                action_cmd = action_cmd/env.max_Iq

            elif args.type == "PPO":
                action_cmd = agent.select_action(self.state)

            self.pub_iq(action_cmd)
            #self.pub_iq(-0.5)


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
