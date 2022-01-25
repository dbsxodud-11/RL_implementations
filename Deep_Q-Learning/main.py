import argparse

import gym
import numpy as np
from tqdm import tqdm

from DQN import DQNAgent
from Double_DQN import DoubleDQNAgent

parser = argparse.ArgumentParser(description='Train Deep Q-Learning Agent in Cartpole Environment')
parser.add_argument('--agent', required=True, default='DQN', help='Choose your RL Agent')
args = parser.parse_args()

env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

num_episodes = 500

if args.agent == "DQN":
    agent = DQNAgent(state_dim, action_dim)
elif args.agent == "Double DQN":
    agent = DoubleDQNAgent(state_dim, action_dim)

for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    episode_loss = []
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward
        transition = [state, action, next_state, reward, done]
        agent.push(transition)
        state = next_state

        if agent.train_start():
            loss = agent.train()
            episode_loss.append(loss)

        if done:
            if agent.train_start():
                agent.write(episode_reward, np.mean(episode_loss))
            break