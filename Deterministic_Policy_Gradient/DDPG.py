# Implementation of DDPG Algorithm
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
wandb.init("DDPG")

from misc import update_model
from building_blocks import Actor, Critic, ReplayMemory


class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim, action_min, action_max,
                 lr=1e-3, gamma=0.99, noise_scale=1.0, noise_scale_decay=0.999, noise_scale_min=0.01, batch_size=120, tau=0.001):
        super(DDPGAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max

        self.lr = lr
        self.gamma = gamma
        self.noise_scale = noise_scale
        self.noise_scale_decay = noise_scale_decay
        self.noise_scale_min = noise_scale_min
        self.batch_size = batch_size
        self.tau = tau
        self.step = 0
        self.decay_step = 10

        self.actor = Actor(self.state_dim, self.action_dim)
        self.target_actor = Actor(self.state_dim, self.action_dim)
        update_model(self.actor, self.target_actor, tau=1.0)

        self.critic = Critic(self.state_dim + self.action_dim, 1)
        self.target_critic = Critic(self.state_dim + self.action_dim, 1)
        update_model(self.critic, self.target_critic, tau=1.0)

        self.mse_loss = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.memory = ReplayMemory(capacity=5000)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state).detach().numpy().squeeze(0) + np.random.normal(0, 1, self.action_dim) * self.noise_scale
        action_norm = np.clip(action, self.action_min, self.action_max)
        return action, action_norm

    def push(self, transition):
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        current_q_values = self.critic(torch.cat((states, actions), dim=1))
        next_q_values = self.target_critic(torch.cat((next_states, self.target_actor(next_states)), dim=1)).detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        value_loss = self.mse_loss(target_q_values, current_q_values)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(torch.cat((states, self.actor(states)), dim=1)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        update_model(self.actor, self.target_actor, tau=self.tau)
        update_model(self.critic, self.target_critic, tau=self.tau)

        if self.step % self.decay_step == 0:
            if self.noise_scale > self.noise_scale_min:
                self.noise_scale *= self.noise_scale_decay
            else:
                self.noise_scale = self.noise_scale_min
        self.step += 1

        return policy_loss.item(), value_loss.item()

    def write(self, reward, policy_loss, value_loss):
        wandb.log({'Reward': reward,
                   'Actor Loss': policy_loss, 'Critic Loss': value_loss})