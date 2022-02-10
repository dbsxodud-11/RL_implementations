import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                                    nn.ReLU(),
                                    nn.Linear(64, output_dim))
        
    def forward(self, x):
        out = self.layers(x)
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(nn.Linear(state_dim + action_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.layers(x)