# Implementation of DDPG Algorithm
import os
import random
import datetime

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def update_model(source, target, tau) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


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


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        states, actions, next_states, rewards, dones = zip(*transitions)

        states = np.vstack(states)
        actions = np.vstack(actions)
        next_states = np.vstack(next_states)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).reshape(-1, 1)

        return states, actions, next_states, rewards, dones
    

class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, action_min, action_max, lr=1e-3, gamma=0.99, noise_scale=1.0, noise_scale_decay=0.999, noise_scale_min=0.01, batch_size=120, tau=0.001):
        super(DDPG, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.lr = lr
        self.gamma = gamma
        self.noise_scale = noise_scale
        self.noise_scale_decay = noise_scale_decay
        self.noise_scale_min = noise_scale_min
        self.batch_size = batch_size
        self.tau = tau
        self.step = 0
        self.decay_step = 10

        self.actor = MLP(self.state_dim, self.action_dim)
        self.target_actor = MLP(self.state_dim, self.action_dim)
        update_model(self.actor, self.target_actor, tau=1.0)

        self.critic = MLP(self.state_dim + self.action_dim, 1)
        self.target_critic = MLP(self.state_dim + self.action_dim, 1)
        update_model(self.critic, self.target_critic, tau=1.0)

        self.mse_loss = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.memory = ReplayMemory(capacity=5000)

        log_dir = './runs/DDPG_Inverted_Pendulum_v0_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer_step = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state).detach().numpy().squeeze(0) + np.random.normal(0, 1, self.action_dim) * self.noise_scale
        action_norm = np.clip(action, self.action_min, self.action_max)
        return action, action_norm

    def push(self, transition):
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)

        current_q_values = self.critic(torch.cat((states, actions), dim=1))
        next_q_values = self.target_critic(torch.cat((next_states, self.target_actor(next_states)), dim=1)).detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        value_loss = self.mse_loss(target_q_values, current_q_values)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(torch.cat((states, self.actor(states)), dim=1)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        update_model(self.actor, self.target_actor, tau=self.tau)
        update_model(self.critic, self.target_critic, tau=self.tau)

        if self.step % self.decay_step == 0:
            if self.noise_scale > self.noise_scale_min:
                self.noise_scale *= self.noise_scale_decay
            else:
                self.noise_scale = self.noise_scale_min
        self.step += 1

        return policy_loss.item(), value_loss.item()

    def write(self, reward, policy_loss, value_loss):
        self.writer.add_scalar('Reward', reward, self.writer_step)
        self.writer.add_scalar('Loss/Actor_Loss', policy_loss, self.writer_step)
        self.writer.add_scalar('Loss/Critic_Loss', value_loss, self.writer_step)
        self.writer.add_scalar('Noise Scale', self.noise_scale, self.step)
        self.writer_step += 1


if __name__ == "__main__":

    env = gym.make('Pendulum-v1') # Continous control task
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_min = env.action_space.low
    action_max = env.action_space.high

    num_episodes = 300
    agent = DDPG(state_dim, action_dim, action_min, action_max)

    for episode in tqdm(range(num_episodes)):
        episode_reward = 0
        episode_actor_loss = []
        episode_critic_loss = []
        state = env.reset()
        for t in range(1000):
            action, action_norm = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_norm)

            episode_reward += reward
            transition = [state, action, next_state, reward, done]
            agent.push(transition)
            state = next_state

            if agent.train_start():
                actor_loss, critic_loss = agent.train()
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
            
            if done:
                if agent.train_start():
                    agent.write(episode_reward, np.mean(episode_actor_loss), np.mean(episode_critic_loss))
                break

    # episode_reward_moving_average = [sum(episode_rewards[max(0, t-10): min(t+10, len(episode_rewards))]) / (min(t+10, len(episode_rewards)) - max(0, t-10) + 1) for t in range(len(episode_rewards))]
    # plt.style.use('ggplot')
    # plt.plot(episode_rewards, linewidth=2.0, color='lightcoral', label='DDPG')
    # plt.plot(episode_reward_moving_average, linewidth=3.0, color='crimson')
    # plt.title('Reward Curve - DDPG')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()