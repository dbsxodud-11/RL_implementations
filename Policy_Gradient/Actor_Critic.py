# Implementation of Actor-Critic Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import wandb
wandb.init('Actor-Critic')

from misc import update_model
from building_blocks import Actor_Continuous, Actor_Discrete, MLP


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None, actor_lr=1e-3, critic_lr=5e-4, gamma=0.99):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continous = continuous
        self.action_min = action_min
        self.action_max = action_max

        self.gamma = gamma
        
        if continuous:
            self.actor_network = Actor_Continuous(state_dim, action_dim)
        else:
            self.actor_network = Actor_Discrete(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_lr)

        self.critic_network = MLP(state_dim, 1)
        self.critic_loss_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continous:
            mu, sigma = self.actor_network(state)
            m = Normal(mu * self.action_max[0], sigma)
            action = m.sample()
            action_prob = m.log_prob(action)
            action = action.detach().numpy()[0]
        else:
            prob = self.actor_network(state)
            m = Categorical(prob)
            action = m.sample()
            action_prob = m.log_prob(action)
            action = action.item()
        return action, action_prob
    
    def train(self, state, action, log_action_prob, reward, next_state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.tensor(action)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)

        current_state_value = self.critic_network(state)
        next_state_value = reward + self.gamma * self.critic_network(next_state)

        critic_loss = self.critic_loss_criterion(current_state_value, next_state_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -log_action_prob * (next_state_value - current_state_value).detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def write(self, reward):
        wandb.log({"Reward": reward})