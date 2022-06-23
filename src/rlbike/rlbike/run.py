import numpy as np
import random
import gym
from collections import deque
import torch
import time
import argparse
from .files import MultiPro
from .files.Agent import Agent
import json
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str, default="HalfCheetahBulletEnv-v0",
                    help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("-per", type=int, default=0, choices=[0, 1],
                    help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("-munchausen", type=int, default=0, choices=[0, 1],
                    help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("-dist", "--distributional", type=int, default=0, choices=[0, 1],
                    help="Using a distributional IQN Critic if set to 1, default=0")
parser.add_argument("-ere", type=int, default=0, choices=[0, 1],
                    help="Adding Emphasizing Recent Experience to the agent if set to 1, default = 0")
parser.add_argument("-n_step", type=int, default=1, help="Using n-step bootstrapping, default=1")
parser.add_argument("-info", type=str, default="rwip", help="Information or name of the run")
parser.add_argument("-d2rl", type=int, choices=[0, 1], default=0,
                    help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("-frames", type=int, default=50000,
                    help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("-eval_every", type=int, default=1000,
                    help="Number of interactions after which the evaluation runs are performed, default = 1000")
parser.add_argument("-eval_runs", type=int, default=3, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("--n_updates", type=int, default=1,
                    help="Update-to-Data (UTD) ratio, updates taken per step with the environment, default=1")
parser.add_argument("-lr_a", type=float, default=3e-4,
                    help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-lr_c", type=float, default=3e-4,
                    help="Critic learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-a", "--alpha", type=float,
                    help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256,
                    help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6),
                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=0.005, help="Softupdate factor tau, default is 0.005")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("-s", "--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel worker, default = 1")
parser.add_argument("-r", "--render_evals", type=int, default=0, choices=[0, 1],
                    help="Rendering the evaluation runs if set to 1, default=0")
parser.add_argument("--trial", type=int, default=0, help="trial")
parser.add_argument("--rep_max", type=int, default=500, help="maximum steps in one episode")
args = parser.parse_args()

env = The_cool_bike()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

state_size = 3
action_size = 1

agent = Agent(state_size=state_size, action_size=action_size, args=args, device=device)


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


def run(args):
    rep_max = args.rep_max

    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    i_episode = 1
    #state = envs.reset(args.saved_model)
    state = np.array([0, 0, 0], dtype=np.float32)
    score = 0
    frames = args.frames // args.worker
    eval_every = args.eval_every // args.worker
    eval_runs = args.eval_runs
    worker = args.worker
    ERE = args.ere
    if ERE:
        episode_K = 0
        eta_0 = 0.996
        eta_T = 1.0
        # episodes = 0
        max_ep_len = 500  # original = 1000
        c_k_min = 2500  # original = 5000

    rep = 0
    for frame in range(1, frames + 1):
        # evaluation runs
        # print("run")
        rep += 1

        #if frame % eval_every == 0 or frame == 1:
        #    evaluate(frame=frame * worker, args=args, eval_runs=eval_runs)

        action = agent.act(state)
        #action = np.clip(action, action_low, action_high) <- no need, already in range (-1,+1)
        #next_state, reward, done, _ = envs.step(action)  # returns np.stack(obs), np.stack(action) ...
        next_state = np.array([0, 0, 0], dtype=np.float32)
        reward = 0
        done = False

        '''
        if frame > frames * 0.8:
            next_state, reward, done, _ = envs.step_q2dot(action)
        else:
            next_state, reward, done, _ = envs.step(action)
        '''
        # print(state, action, reward, next_state, done)
        # for s, a, r, ns, d in zip(state, action, reward, next_state, done):
        #    agent.step(s, a, r, ns, d, frame, ERE)
        agent.step(state, action, reward, next_state, [done], frame, ERE)

        print(time.time())

        if ERE:
            eta_t = eta_0 + (eta_T - eta_0) * (frame / (frames + 1))
            episode_K += 1
        state = next_state
        score += np.mean(reward)

        # if done.any():
        if done or rep >= rep_max:
            rep = 0
            if ERE:
                for k in range(1, episode_K):
                    c_k = max(int(agent.memory.__len__() * eta_t ** (k * (max_ep_len / episode_K))), c_k_min)
                    agent.ere_step(c_k)
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            #writer.add_scalar("Average100", np.mean(scores_window), frame * worker)
            print('\rEpisode {}\tFrame: [{}/{}]\t Reward: {:.2f} \tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker, frames, score, np.mean(scores_window)), end="", flush=True)
            # if i_episode % 100 == 0:
            #    print('\rEpisode {}\tFrame \tReward: {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, round(eval_reward,2), np.mean(scores_window)), end="", flush=True)
            i_episode += 1
            #state = envs.reset(args.saved_model)
            state = np.array([0, 0, 0], dtype=np.float32)
            score = 0
            episode_K = 0


class Node_RL(Node):

    def __init__(self):
        super().__init__('node_rl')
        self.imu_sub = self.create_subscription(
            Float64, 'list_deg', self.imu_callback, 10)
        self.imu_sub  # prevent unused variable warning

        self.motor_sub = self.create_subscription(
            Float64, 'speed_feedback', self.imu_callback, 10)
        self.motor_sub  # prevent unused variable warning

    def imu_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

    def imu_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


#if __name__ == "__main__":
def main(args=args):
    rclpy.init(args=args)
    node_rl = Node_RL()

    #if args.saved_model == None:
    #    writer = SummaryWriter("runs_v3/" + args.info + str(args.trial))
    # envs = MultiPro.SubprocVecEnv([lambda: gym.make(args.env) for i in range(args.worker)])
    # eval_env = gym.make(args.env)

    #envs = Pendulum(args.render_evals, args.seed)
    #eval_env = Pendulum(args.render_evals, args.seed + 1)
    t0 = time.time()
    print("###########################")

    if args.saved_model != None:
        agent.actor_local.load_state_dict(torch.load(args.saved_model, map_location=device))
        evaluate(frame=None, args=args, capture=False)
    else:
        run(args)
        t1 = time.time()
        timer(t0, t1)

        # save policy
        torch.save(agent.actor_local.state_dict(),
                   '/home/nvidia/ros2_rlbike/runs_v3/{}{}/'.format(args.info, args.trial) + args.info + str(args.trial) + ".pth")

        # save parameter
        with open('/home/nvidia/ros2_rlbike/runs_v3/{}{}/'.format(args.info, args.trial) + args.info + str(args.trial) + ".json", 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    #eval_env.close()
    #if args.saved_model == None:
    #    writer.close()

    rclpy.spin(node_rl)
    node_rl.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()