import argparse

import gym
import numpy as np
from tqdm import tqdm

from DDPG import DDPGAgent

parser = argparse.ArgumentParser()
parser.add_argument('--agent', required=True, default='DDPG', help='Choose your RL Agent')
parser.add_argument('--env', required=True, default="Pendulum", help="Choose Environment", choices=["Pendulum"])
args = parser.parse_args()

if args.env == "Pendulum":
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_min = env.action_space.low
    action_max = env.action_space.high

num_episodes = 500

if args.agent == "DDPG":
    agent = DDPGAgent(state_dim, action_dim, action_min, action_max)

for episode in tqdm(range(num_episodes)):
    episode_reward = 0
    episode_policy_loss = []
    episode_value_loss = []

    state = env.reset()
    done = False
    while not done:
        action, action_norm = agent.select_action(state)
        next_state, reward, done, _ = env.step(action_norm)

        episode_reward += reward
        transition = [state, action, next_state, reward, done]
        agent.push(transition)
        state = next_state

        if agent.train_start():
            policy_loss, value_loss = agent.train()
            episode_policy_loss.append(policy_loss)
            episode_value_loss.append(value_loss)

        if done:
            if agent.train_start():
                agent.write(episode_reward, np.mean(episode_policy_loss), np.mean(episode_value_loss))
            break