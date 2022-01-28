# Implementation of PER(Prioritized Experience Replay) Algorithm
from contextlib import redirect_stderr
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
wandb.init('Cartpole')

from misc import update_model
from building_blocks import MLP, PER
from DQN import DQNAgent

class PERAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=120, eps=0.99, eps_decay=0.999, eps_threshold=0.01, tau=0.001):
        super(PERAgent, self).__init__(state_dim, action_dim, lr, gamma, batch_size, eps, eps_decay, eps_threshold)
        
        self.tau = tau

        self.criterion = nn.MSELoss(reduction='none')

        self.memory = PER(capacity=5000)

    def push(self, transition):
        state, action, next_state, reward, done = transition

        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.tensor(action, dtype=torch.int64).reshape(-1, 1)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32).reshape(-1, 1)
        done = torch.tensor(done, dtype=torch.float32).reshape(-1, 1)

        next_action = torch.max(self.main_network(next_state), dim=1)[1].reshape(-1, 1)
        target_q_value = reward + self.gamma * self.target_network(next_state).gather(1, next_action) * (1 - done)

        td_error = torch.abs(self.main_network(state).gather(1, action) - target_q_value).detach().numpy()

        self.memory.push(transition, td_error)

    def train(self):
        batch, idxs, is_weight = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = batch

        current_q_values = self.main_network(states).gather(1, actions)
        # Mitigate Maximization Bias
        next_actions = torch.max(self.main_network(next_states), dim=1)[1].reshape(-1, 1)
        next_q_values = self.target_network(next_states).gather(1, next_actions) * (1 - dones)
        target_q_values = rewards + self.gamma * next_q_values.detach()

        mse_loss = self.criterion(target_q_values, current_q_values)
        weighted_loss = (is_weight * mse_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Update Transition Priority
        errors = torch.abs(target_q_values - current_q_values).detach().numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors[i])

        self.step += 1
        if self.step % self.update_step:
            # Soft Update
            update_model(self.main_network, self.target_network, tau=self.tau)
            if self.eps > self.eps_threshold:
                self.eps *= self.eps_decay
            else:
                self.eps = self.eps_threshold

        return weighted_loss.item()
