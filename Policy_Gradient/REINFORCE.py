# Implementation of REINFORCE Algorithm
from os import access
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import wandb

from misc import update_model
from building_blocks import MLP, Actor_Continuous, Actor_Discrete


class REINFORCE(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None, baseline=False, lr=1e-3, gamma=0.99):
        super(REINFORCE, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continous = continuous
        self.action_min = action_min
        self.action_max = action_max
        self.baseline = baseline

        self.lr = lr
        self.gamma = gamma
        
        if continuous:
            self.main_network = Actor_Continuous(state_dim, action_dim)
        else:
            self.main_network = Actor_Discrete(state_dim, action_dim)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)

        if baseline:
            self.value_network = MLP(state_dim, 1)
            self.mse_loss = nn.MSELoss()
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)
        
        self.buffer = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continous:
            mu, sigma = self.main_network(state)
            m = Normal(mu * self.action_max[0], sigma)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        else:
            prob = self.main_network(state)
            m = Categorical(prob)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def update_policy(self):
        states, actions, rewards = zip(*self.buffer)
        states = np.stack(states)
        states = torch.FloatTensor(states)
        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        returns = [np.sum(rewards[i:] * (self.gamma ** np.arange(len(rewards) - i))) for i in range(len(rewards))]
        returns = torch.FloatTensor(returns)
        
        if self.continous:
            mu, sigma = self.main_network(states)
            m = Normal(mu * self.action_max[0], sigma)
        else:
            probs = self.main_network(states)
            m = Categorical(probs)

        if self.baseline:
            baselines = self.value_network(states).squeeze()
            value_loss = self.mse_loss(returns, baselines)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            loss = -torch.mean(m.log_prob(actions) * (returns - baselines.detach()))
        else:
            loss = -torch.mean(m.log_prob(actions) * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def train(self, env, num_episodes):
        for _ in tqdm(range(num_episodes)):
            episode_reward = 0
            self.buffer.clear()

            state = env.reset()
            done = False
            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                self.buffer.append((state, action, reward))

                episode_reward += reward
                state = next_state

                if done:
                    self.update_policy()
                    wandb.log({"Reward": episode_reward})
                    break

    def __str__(self):
        if self.baseline:
            return "REINFORCE with Baseline"
        else:
            return "REINFORCE"
