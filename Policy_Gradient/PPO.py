# Implementation of PPO Algorithm
import random

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import wandb

from misc import update_model
from building_blocks import Actor_Continuous, Actor_Discrete, MLP


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None,
                 actor_lr=1e-3, critic_lr=5e-4, gamma=0.99, trade_off=0.99, eps=0.2):
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
        self.mse_loss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.trade_off = trade_off
        self.eps = eps

        self.buffer = []
        self.train_step = 10

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

    def update_policy(self):
        total_policy_loss = 0 
        total_value_loss = 0
        
        states, actions, action_log_probs, rewards = zip(*self.buffer)

        states = np.stack(states)
        states = torch.FloatTensor(states)
        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        action_log_probs = np.array(action_log_probs)
        action_log_probs = torch.FloatTensor(action_log_probs)
        returns = [np.sum(rewards[i:] * (self.gamma ** np.arange(len(rewards) - i))) for i in range(len(rewards))]
        returns = torch.FloatTensor(returns).reshape(-1, 1)

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
        
        for _ in range(self.train_step):
                    
            #get new log prob of actions for all input states
            probs = self.actor(states)
            m = Categorical(probs) 
            new_action_log_probs = m.log_prob(actions)
            
            policy_ratio = (new_action_log_probs - action_log_probs.detach()).exp()
                    
            policy_loss_1 = policy_ratio * advantage_values
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.eps, max = 1.0 + self.eps) * advantage_values
            
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
            
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            values = self.critic(states)
            value_loss = F.smooth_l1_loss(returns, values).sum()
        
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return total_policy_loss / self.train_step, total_value_loss / self.train_step

    def train(self, env, num_episodes):
        for _ in tqdm(range(num_episodes)):
            episode_reward = 0
            self.buffer.clear()

            state = env.reset()
            done = False
            while not done:
                action, action_log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                self.buffer.append((state, action, action_log_prob, reward))

                episode_reward += reward
                state = next_state

                if done:
                    self.update_policy()
                    wandb.log({"Reward": episode_reward})
                    break


    def __str__(self):
        return "PPO"