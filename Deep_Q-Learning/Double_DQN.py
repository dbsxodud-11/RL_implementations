# Implementation of Double DQN Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
wandb.init('Cartpole')

from misc import update_model
from building_blocks import MLP, ReplayMemory

class DoubleDQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=120, eps=0.99, eps_decay=0.999, eps_threshold=0.01, tau=0.001):
        super(DoubleDQNAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_threshold = eps_threshold
        self.tau = tau

        self.main_network = MLP(self.state_dim, self.action_dim)
        self.target_network = MLP(self.state_dim, self.action_dim)
        update_model(self.main_network, self.target_network, tau=1.0)

        self.step = 0
        self.update_step = 1000

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)

        self.memory = ReplayMemory(capacity=5000)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if random.random() < self.eps:
            action = np.random.randint(self.action_dim)
        else:
            action = torch.argmax(self.main_network(state)).item()
        return action

    def push(self, transition):
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        current_q_values = self.main_network(states).gather(1, actions)
        # Mitigate Maximization Bias
        next_actions = torch.max(self.main_network(next_states), dim=1)[1].reshape(-1, 1)
        next_q_values = self.target_network(next_states).gather(1, next_actions) * (1 - dones)
        target_q_values = rewards + self.gamma * next_q_values.detach()

        mse_loss = self.criterion(target_q_values, current_q_values)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % self.update_step:
            # Soft Update
            update_model(self.main_network, self.target_network, tau=self.tau)
            if self.eps > self.eps_threshold:
                self.eps *= self.eps_decay
            else:
                self.eps = self.eps_threshold

        return mse_loss.item()

    def write(self, reward, loss):
        wandb.log({"Reward": reward, 
                   "Loss": loss,
                   "Epsilon": self.eps})
