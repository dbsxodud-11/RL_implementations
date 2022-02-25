import random

import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, output_dim))
    
    def forward(self, x):
        out = self.layers(x)
        return out


class Actor_Discrete(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_Discrete, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, output_dim))
    
    def forward(self, x):
        out = self.layers(x)
        out = F.softmax(out, dim=-1)
        return out
    

class Actor_Continuous(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_Continuous, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU())

        self.mu_head = nn.Linear(64, output_dim)
        self.sigma_head = nn.Linear(64, output_dim)
        
    def forward(self, x):
        out = self.layers(x)

        mu = torch.tanh(self.mu_head(out))
        sigma = F.softplus(self.sigma_head(out))
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(nn.Linear(state_dim + action_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.layers(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(ValueNetwork, self).__init__()

        self.model = nn.Sequential(nn.Linear(state_dim, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, output_dim))

    def forward(self, x):
        return self.model(x)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.model1 = nn.Sequential(nn.Linear(state_dim + action_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

        self.model2 = nn.Sequential(nn.Linear(state_dim + action_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model1(x), self.model2(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        states, actions, action_probs, rewards, next_states, dones = zip(*transitions)

        states = np.vstack(states)
        next_states = np.vstack(next_states)
        action_probs = np.vstack(action_probs)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).reshape(-1, 1)
        action_probs = torch.tensor(action_probs, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(-1, 1)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).reshape(-1, 1)

        return states, actions, action_probs, rewards, next_states, dones