# Implementation of Actor-Critic Algorithm
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continous = continuous
        self.action_min = action_min
        self.action_max = action_max

        self.gamma = gamma
        
        if continuous:
            self.actor_network = Actor_Continuous(state_dim, action_dim)
        else:
            self.actor_network = Actor_Discrete(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_lr)

        self.critic_network = MLP(state_dim, 1)
        self.critic_loss_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        self.buffer = []
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.continous:
            mu, sigma = self.actor_network(state)
            m = Normal(mu * self.action_max[0], sigma)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        else:
            prob = self.actor_network(state)
            m = Categorical(prob)
            action = m.sample()
            action_log_prob = m.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def update_policy(self):
        states, actions, rewards, next_states, dones = zip(*self.buffer)

        states = np.stack(states)
        states = torch.FloatTensor(states)
        actions = np.array(actions)
        actions = torch.FloatTensor(actions)
        rewards = np.array(rewards)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        dones = np.array(dones)
        dones = torch.tensor(dones, dtype=torch.int64).reshape(-1, 1)
        next_states = np.stack(next_states)
        next_states = torch.FloatTensor(next_states)

        pred_state_values = self.critic_network(states)
        target_state_values = rewards + self.gamma * self.critic_network(next_states) * (1 - dones)

        critic_loss = self.critic_loss_criterion(pred_state_values, target_state_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.continous:
            mu, sigma = self.actor_network(states)
            m = Normal(mu * self.action_max[0], sigma)
        else:
            probs = self.actor_network(states)
            m = Categorical(probs)

        actor_loss = -(m.log_prob(actions).view(-1, 1) * (target_state_values - pred_state_values).detach()).sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, env, num_episodes):
        for _ in tqdm(range(num_episodes)):
            episode_reward = 0
            self.buffer.clear()
            state = env.reset()
            done = False
            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.buffer.append((state, action, reward, next_state, done))

                episode_reward += reward
                state = next_state

                if done:
                    self.update_policy()
                    wandb.log({"Reward": episode_reward})
                    break

    def __str__(self):
        return "Actor-Critic"