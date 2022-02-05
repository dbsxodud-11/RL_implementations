# Implementation of REINFORCE Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import wandb
wandb.init('Pendulum')

from misc import update_model
from building_blocks import MLP_Continuous, MLP_Discrete


class REINFORCE(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None, lr=1e-3, gamma=0.99):
        super(REINFORCE, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continous = continuous
        self.action_min = action_min
        self.action_max = action_max

        self.lr = lr
        self.gamma = gamma
        
        if continuous:
            self.main_network = MLP_Continuous(state_dim, action_dim)
        else:
            self.main_network = MLP_Discrete(state_dim, action_dim)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continous:
            mu, sigma = self.main_network(state)
            m = Normal(mu * self.action_max[0], sigma)
            action = m.sample().detach().numpy()[0]
        else:
            prob = self.main_network(state)
            m = Categorical(prob)
            action = m.sample().item()
        return action
    
    def train(self, states, actions, rewards):
        states = np.stack(states)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        returns = [np.sum(rewards[i:] * (self.gamma ** np.array(range(i, len(rewards))))) for i in range(len(rewards))]
        returns = torch.FloatTensor(returns)
        
        if self.continous:
            mu, sigma = self.main_network(states)
            m = Normal(mu * self.action_max[0], sigma)
        else:
            probs = self.main_network(states)
            m = Categorical(probs)
        loss = -torch.mean(m.log_prob(actions) * returns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def write(self, reward):
        wandb.log({"Reward": reward})

        
        
        
        
        