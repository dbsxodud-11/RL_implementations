import random

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from misc import SumTree

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
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        states, actions, next_states, rewards, dones = zip(*transitions)

        states = np.vstack(states)
        next_states = np.vstack(next_states)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).reshape(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).reshape(-1, 1)

        return states, actions, next_states, rewards, dones


class PER:
    def __init__(self, capacity, eps=0.01, alpha=0.6, beta=0.4):
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 0.001

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def push(self, sample, error):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = torch.tensor(is_weight, dtype=torch.float32).reshape(-1, 1)

        states, actions, next_states, rewards, dones = zip(*batch)

        states = np.vstack(states)
        next_states = np.vstack(next_states)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).reshape(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).reshape(-1, 1)

        batch = [states, actions, next_states, rewards, dones]
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)