# Implementation of REINFORCE Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb
wandb.init('Pendulum')

from misc import update_model
from building_blocks import MLP


class REINFORCE(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        super(REINFORCE, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr = lr
        self.gamma = gamma
        
        self.main_network = MLP(state_dim, action_dim)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
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
        
        probs = self.main_network(states)
        m = Categorical(probs)
        loss = -torch.mean(m.log_prob(actions) * returns)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def write(self, reward):
        wandb.log({"Reward": reward})

        
        
        
        
        