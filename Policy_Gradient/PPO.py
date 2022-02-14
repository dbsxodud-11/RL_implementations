# Implementation of PPO Algorithm
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from misc import update_model
from building_blocks import Actor_Continuous, Actor_Discrete, MLP


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None,
                 actor_lr=1e-3, critic_lr=5e-4, gamma=0.99, eps=0.2):
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

        self.gamma = gamma
        self.eps = eps

        self.step = 0
        self.train_step = 100

        self.state_list = []
        self.action_list = []
        self.action_log_prob_list = []
        self.reward_list = []
        self.next_state_list = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continuous:
            mu, sigma =  self.actor(state)
            m = Normal(mu * self.action_max[0], sigma)
            action = m.sample()
            action_log_prob = m.log_prob(action)
            action = torch.clamp(action, self.action_min[0], self.action_max[0])
        else:
            prob = self.actor(state)
            m = Categorical(prob)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        return action.item(), action_log_prob.item()

    def train(self, state, action, action_log_prob, reward, next_state):
        self.state_list.append(state)
        self.action_list.append(action)
        self.action_log_prob_list.append(action_log_prob)
        self.reward_list.append((reward + 8) / 8)
        # self.reward_list.append(reward)
        self.next_state_list.append(next_state)

        self.step += 1
        if self.step % self.train_step == 0:

            states = np.stack(self.state_list)
            states = torch.FloatTensor(states)

            actions = np.array(self.action_list)
            actions = torch.FloatTensor(actions).reshape(-1, 1)

            old_action_log_probs = np.array(self.action_log_prob_list)
            old_action_log_probs = torch.FloatTensor(old_action_log_probs).reshape(-1, 1)

            rewards = np.array(self.reward_list)
            rewards = torch.FloatTensor(rewards).reshape(-1, 1)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            next_states = np.stack(self.next_state_list)
            next_states = torch.FloatTensor(next_states)

            advantage_values = rewards + self.gamma * self.critic(next_states) - self.critic(states)

            for _ in range(10):
               for idx in BatchSampler(
                    SubsetRandomSampler(range(100)), 32, False):
                    if self.continuous:
                        new_mu, new_sigma = self.actor(states[idx])
                        m = Normal(new_mu * self.action_max[0], new_sigma)
                    else:
                        new_prob = self.actor(states[idx])
                        m = Categorical(new_prob)
                    
                    action_log_prob = m.log_prob(actions[idx])
                    ratio = torch.exp(action_log_prob - old_action_log_probs[idx])

                    surr1 = ratio * advantage_values[idx].detach()
                    surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantage_values[idx].detach()
                    policy_loss = -torch.min(surr1, surr2).mean()

                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    value = self.critic(states[idx])
                    target = rewards[idx] + self.gamma * self.critic(next_states[idx]).detach()
                    # value_loss = F.smooth_l1_loss(value, target)
                    value_loss = self.critic_criterion(value, target)

                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()

            self.state_list = []
            self.action_list = []
            self.action_log_prob_list = []
            self.reward_list = []
            self.next_state_list = []
