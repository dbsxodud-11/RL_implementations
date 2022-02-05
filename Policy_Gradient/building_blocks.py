import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Discrete(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Discrete, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, output_dim))
    
    def forward(self, x):
        out = self.layers(x)
        out = F.softmax(out, dim=-1)
        return out
    

class MLP_Continuous(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Continuous, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.ReLU())
        
        self.mu_head = nn.Linear(64, output_dim)
        self.sigma_head = nn.Linear(64, output_dim)
        
    def forward(self, x):
        out = self.layers(x)
        mu = torch.tanh(self.mu_head(out))
        sigma = F.softplus(self.sigma_head(out))
        return mu, sigma