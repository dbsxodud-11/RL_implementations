import argparse

import gym
import numpy as np
from tqdm import tqdm

from REINFORCE import REINFORCE

parser = argparse.ArgumentParser(description='Train Policy Gradient Agent in Cartpole Environment')
parser.add_argument('--agent', required=True, default='REINFORCE', help='Choose your RL Agent')
parser.add_argument('--env', required=True, default="Cartpole", help="Choose Environment", choices=["Cartpole", "Pendulum"])
args = parser.parse_args()

continuous = False
action_min = None
action_max = None

if args.env == "Cartpole":
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
elif args.env == "Pendulum":
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

for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    state_list = []
    action_list = []
    reward_list = []

    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        state_list.append(state)
        action_list.append(action)
        reward_list.append(reward)

        episode_reward += reward
        state = next_state

        if done:
            agent.train(state_list, action_list, reward_list)
            agent.write(episode_reward)
            break