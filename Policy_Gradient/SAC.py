# Implementation of SAC Algorithm
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
from building_blocks import Actor_Continuous, Actor_Discrete, ValueNetwork, QNetwork, ReplayMemory


class SAC(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=False, action_min=None, action_max=None,
                 actor_lr=1e-3, critic_lr=5e-4, batch_size=64, alpha=0.05, gamma=0.99, tau=0.01):
        super(SAC, self).__init__()

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

        self.value_fn = ValueNetwork(state_dim, 1)
        self.target_value_fn = ValueNetwork(state_dim, 1)
        update_model(self.value_fn, self.target_value_fn, tau=1.0)
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=critic_lr)
    
        self.q_fn = QNetwork(state_dim, action_dim)
        self.q_optimizer = optim.Adam(self.q_fn.parameters(), lr=critic_lr)

        self.mse_loss = nn.MSELoss()

        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayMemory(capacity=5000)
        self.step = 1
        self.train_step = 1

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
        return action.item(), prob.detach().numpy()

    def push(self, transition):
        self.memory.push(transition)

    def update_start(self):
        return len(self.memory) >= self.batch_size and self.step % self.train_step == 0

    def update(self):
        
        states, actions, action_probs, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # 1. Value function Update
        pred_values = self.value_fn(states)
        probs = self.actor(states)
        m = Categorical(probs)
        pred_actions = m.sample()
        log_probs = m.log_prob(pred_actions)

        q_values1, q_values2 = self.q_fn(states, probs)
        q_values = torch.min(q_values1, q_values2)

        value_loss = self.mse_loss(pred_values, (q_values - self.alpha * log_probs.reshape(-1, 1)).detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 2. Q function update
        pred_q_values1, pred_q_values2 = self.q_fn(states, action_probs)
        target_q_values = rewards + self.gamma * self.target_value_fn(next_states) * (1 - dones)

        q_loss1 = self.mse_loss(pred_q_values1, target_q_values.detach())
        q_loss2 = self.mse_loss(pred_q_values2, target_q_values.detach())
        q_loss = q_loss1 + q_loss2
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 3. Policy Update
        probs = self.actor(states)
        m = Categorical(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions)

        q_values1, q_values2 = self.q_fn(states, probs)
        q_values = torch.min(q_values1, q_values2)

        policy_loss = (self.alpha * log_probs.view(-1, 1) - q_values).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 4. Target Update
        update_model(self.value_fn, self.target_value_fn, tau=self.tau)

        return value_loss, q_loss, policy_loss

    def train(self, env, num_episodes):
        for episode in tqdm(range(num_episodes)):
            episode_reward = 0

            state = env.reset()
            done = False
            while not done:
                action, action_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                self.push((state, action, action_prob, reward, next_state, done))

                if self.update_start():
                    value_loss, q_loss, policy_loss = self.update()
                    wandb.log({"Value Loss": value_loss,
                                "Q Loss": q_loss,
                                "Policy Loss": policy_loss}, commit=False)

                episode_reward += reward
                state = next_state
                self.step += 1

                if done:
                    wandb.log({"Reward": episode_reward}, step=episode,  commit=False)
                    break

    def __str__(self):
        return "SAC"
