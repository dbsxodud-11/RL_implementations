# Implementation of A2C Algorithm
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


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[64 for _ in range(2)]):
        super(Actor, self).__init__()

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


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[64 for _ in range(2)]):
        super(Critic, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [input_dim] + hidden
        output_dims = hidden + [output_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))
        self.layers.append(nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class A2C(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, entropy_coef=0.5):
        super(A2C, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, 1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.log_prob_list = []
        self.entropy_list = []
        self.value_list = []


    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_prob_list.append(-m.log_prob(action))
        self.entropy_list.append(m.entropy())
        self.value_list.append(self.critic(state))

        return action.item()

    def train(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        actor_loss = []
        for log_prob, R, v, E in zip(self.log_prob_list, returns, self.value_list, self.entropy_list):
            actor_loss.append(log_prob * (R - v.detach()) + self.entropy_coef * E)
        actor_loss = torch.cat(actor_loss).sum()

        critic_loss = []
        for R, v in zip(returns, self.value_list):
            critic_loss.append((R - v)**2)
        critic_loss = torch.cat(critic_loss).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.log_prob_list = []
        self.entropy_list = []
        self.value_list = []


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = 300

    agent = A2C(state_dim, action_dim)

    total_reward = []
    for episode in range(num_episodes):
        episode_reward = []
        state = env.reset()
        for t in range(1000):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward.append(reward)
            if done:
                total_reward.append(sum(episode_reward))
                print(f"Episode {episode+1}: finished after {t+1} timesteps")
                agent.train(episode_reward)
                break
        
    moving_average = [sum(total_reward[max(0, t-10): min(t+10, len(total_reward))]) / (min(t+10, len(total_reward)) - max(0, t-10) + 1) for t in range(len(total_reward))]
    plt.style.use('ggplot')
    plt.plot(total_reward, linewidth=2.0, color='lightcoral', label='A2C')
    plt.plot(moving_average, linewidth=3.0, color='crimson')
    plt.title('Training Curve - A2C')
    plt.legend()
    plt.tight_layout()
    plt.show()
