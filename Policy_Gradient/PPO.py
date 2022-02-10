# Implementation of PPO Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import wandb
wandb.init('PPO')

from misc import update_model
from building_blocks import Actor_Continuous, Actor_Discrete, MLP


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, action_min=None, action_max=None, continuous=False,
                 actor_lr=1e-3, critic_lr=1e-3, sigma=0.6, sigma_decay=0.995, sigma_min=0.01, gamma=0.99, trade_off=0.99, eps=0.2):
        super(PPO, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.continuous = continuous

        if self.continuous:
            self.actor = Actor_Continuous(state_dim, action_dim)
        else:
            self.actor = Actor_Discrete(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = MLP(state_dim, 1)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min

        self.gamma = gamma
        self.trade_off = trade_off

        self.eps = eps

        self.step = 0
        self.train_step = 20

        self.states_list = []
        self.actions_list = []
        self.log_probs_list = []
        self.returns_list = []
        self.values_list = []
        self.advantages_list = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continuous:
            mu =  self.actor(state)
            m = Normal(mu * self.action_max[0], self.sigma)
            action = m.sample().detach().numpy()[0]
        else:
            prob = self.actor(state)
            m = Categorical(prob)
            action = m.sample().item()
        return action

    def train(self, states, actions, rewards):
        states = np.stack(states)
        states = torch.FloatTensor(states)
        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        returns = [np.sum(rewards[i:] * (self.gamma ** np.arange(len(rewards) - i))) for i in range(len(rewards))]
        returns = torch.FloatTensor(returns).reshape(-1, 1)
        returns = (returns - returns.mean()) / returns.std()

        if self.continuous:
            mu = self.actor(states)
            m = Normal(mu * self.action_max[0], self.sigma)
        else:
            probs = self.actor(states)
            m = Categorical(probs)

        values = self.critic(states)
        next_value = 0.0
        advantage_value = 0.0
        advantage_values = []
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + self.gamma * next_value - v
            next_value = v
            advantage_value = td_error + advantage_value * self.gamma * self.trade_off
            advantage_values.insert(0, advantage_value)
        advantage_values = torch.FloatTensor(advantage_values)

        self.states_list.append(states)
        self.actions_list.append(actions)
        self.log_probs_list.append(m.log_prob(actions))
        self.returns_list.append(returns)
        self.values_list.append(values)
        self.advantages_list.append(advantage_values)
        
        self.step += 1
        if self.step % self.train_step == 0:
            for states, actions, log_probs, returns, values, advantages in zip(self.states_list, self.actions_list, self.log_probs_list, self.returns_list, self.values_list, self.advantages_list):
                if self.continuous:
                    new_mu = self.actor(states)
                    m = Normal(new_mu * self.action_max[0], self.sigma)
                else:
                    new_probs = self.actor(states)
                    m = Categorical(new_probs)
                new_log_probs = m.log_prob(actions)
                likelihood_ratio = (new_log_probs - log_probs.detach()).exp()

                policy_loss = -torch.min((likelihood_ratio * advantages).mean(), (torch.clamp(likelihood_ratio, 1 - self.eps, 1 + self.eps) * advantages).mean())
                value_loss = self.critic_criterion(values, returns)

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

            self.states_list = []
            self.actions_list = []
            self.log_probs_list = []
            self.returns_list = []
            self.values_list = []
            self.advantages_list = []

            if self.sigma <= self.sigma_min:
                self.sigma = self.sigma_min
            else:
                self.sigma *= self.sigma_decay
        
    def write(self, reward):
        wandb.log({"Reward": reward, "Sigma": self.sigma})