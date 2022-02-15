import numpy as np
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