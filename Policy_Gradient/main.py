import argparse
import datetime

import gym
import wandb
import numpy as np
from tqdm import tqdm

from REINFORCE import REINFORCE
from Actor_Critic import ActorCritic
from GAE import GAE
from PPO import PPO
from SAC import SAC


parser = argparse.ArgumentParser(description='Train Policy Gradient Agent in Cartpole Environment')
parser.add_argument('--agent', required=True, default='REINFORCE', help='Choose your RL Agent')
parser.add_argument('--env', required=True, default="Cartpole", help="Choose Environment", choices=["Cartpole", "Pendulum"])
args = parser.parse_args()

continuous = False
action_min = None
action_max = None

env_name = args.env
if env_name == "Cartpole":
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
elif env_name == "Pendulum":
    env = gym.make('Pendulum-v1')
    continuous = True
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_min = env.action_space.low
    action_max = env.action_space.high

num_episodes = 1000

if args.agent == "REINFORCE":
    agent = REINFORCE(state_dim, action_dim, continuous, action_min, action_max)
elif args.agent == "REINFORCE with baseline":
    agent = REINFORCE(state_dim, action_dim, continuous, action_min, action_max, baseline=True)
elif args.agent == "Actor-Critic":
    agent = ActorCritic(state_dim, action_dim, continuous, action_min, action_max)
elif args.agent == "GAE":
    agent = GAE(state_dim, action_dim, continuous, action_min, action_max, gamma=0.99, trade_off=0.99)
elif args.agent == "PPO":
    agent = PPO(state_dim, action_dim, continuous, action_min, action_max, gamma=0.99, eps=0.2)
elif args.agent == "SAC":
    agent = SAC(state_dim, action_dim, continuous, action_min, action_max, alpha=0.05, gamma=0.99)

time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
wandb.init(project=f"Policy Gradient Algorithms - {env}", name=f"{agent}: {time}")

agent.train(env, num_episodes)