# Implementation of REINFORCE Algorithm
import os
import random

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[64 for _ in range(2)]):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [input_dim] + hidden
        output_dims = hidden + [output_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=1)
        

class REINFORCE(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        super(REINFORCE, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.policy_network = MLP(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

        self.total_reward = []
        self.log_probs = []
        self.returns = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def train(self, rewards):
        self.total_reward.append(sum(rewards))
        
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.log_probs = []


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = 500

    agent = REINFORCE(state_dim, action_dim)

    for episode in range(num_episodes):
        episode_rewards = []
        state = env.reset()
        for t in range(1000):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                agent.train(episode_rewards)
                break
        
    total_reward = agent.total_reward
    moving_average = [sum(total_reward[max(0, t-10): min(t+10, len(total_reward))]) / (min(t+10, len(total_reward)) - max(0, t-10) + 1) for t in range(len(total_reward))]
    plt.style.use('ggplot')
    plt.plot(total_reward, linewidth=2.0, color='lightcoral', label='REINFORCE')
    plt.plot(moving_average, linewidth=3.0, color='crimson')
    plt.title('Training Curve - REINFORCE')
    plt.legend()
    plt.tight_layout()
    plt.show()