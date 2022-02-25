# Implementation of GAE Algorithm
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import wandb

from misc import update_model
from building_blocks import Actor_Continuous, Actor_Discrete, MLP


class GAE(nn.Module):
    def __init__(self, state_dim, action_dim, action_min=None, action_max=None, continuous=False,
                 actor_lr=1e-3, critic_lr=1e-3, sigma=0.5, gamma=0.99, trade_off=0.99):
        super(GAE, self).__init__()

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

        self.gamma = gamma
        self.trade_off = trade_off

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continuous:
            mu, sigma = self.actor(state)
            m = Normal(mu * self.action_max[0], sigma)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        else:
            prob = self.actor(state)
            m = Categorical(prob)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        return action.item(), action_log_prob.item()

    def update_policy(self, states, actions, rewards):
        states = np.stack(states)
        states = torch.FloatTensor(states)
        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        returns = [np.sum(rewards[i:] * (self.gamma ** np.arange(len(rewards) - i))) for i in range(len(rewards))]
        returns = torch.FloatTensor(returns).reshape(-1, 1)
        
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
        
        policy_loss = -(m.log_prob(actions) * advantage_values).mean()
        value_loss = self.critic_criterion(values, returns)
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    def train(self, env, num_episodes):
        for _ in tqdm(range(num_episodes)):
            episode_reward = 0
            state_list = []
            action_list = []
            reward_list = []

            state = env.reset()
            done = False
            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                state_list.append(state)
                action_list.append(action)
                reward_list.append(reward)

                episode_reward += reward
                state = next_state

                if done:
                    self.update_policy(state_list, action_list, reward_list)
                    wandb.log({"Reward": episode_reward})
                    break
        
    def __str__(self):
        return "GAE"