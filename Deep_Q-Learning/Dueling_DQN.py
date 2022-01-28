# Implementation of Dueling DQN Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
wandb.init('Cartpole')

from misc import update_model
from building_blocks import DuelingNet, ReplayMemory
from DQN import DQNAgent

class DuelingDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=120, eps=0.99, eps_decay=0.999, eps_threshold=0.01):
        super(DuelingDQNAgent, self).__init__(state_dim, action_dim, lr, gamma, batch_size, eps, eps_decay, eps_threshold)

        self.main_network = DuelingNet(state_dim, action_dim)
        self.target_network = DuelingNet(state_dim, action_dim)
        update_model(self.main_network, self.target_network, tau=1.0)
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)