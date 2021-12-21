# Implementation of DQN Algorithm
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
        next_states = np.vstack(next_states)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).reshape(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).reshape(-1, 1)

        return states, actions, next_states, rewards, dones


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=120, eps=0.9, eps_decay=0.999, eps_threshold=0.1, tau=0.001):
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_threshold = eps_threshold

        self.main_network = MLP(self.state_dim, self.action_dim)
        self.target_network = MLP(self.state_dim, self.action_dim)
        update_model(self.main_network, self.target_network, tau=1.0)
        self.tau = tau
        self.step = 0
        self.update_step = 1000

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)

        self.memory = ReplayMemory(capacity=5000)

        log_dir = './runs/DQN_CartPole_v0_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer_step = 0

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if random.random() < self.eps:
            action = np.random.randint(self.action_dim)
        else:
            action = torch.argmax(self.main_network(state)).item()
        return action

    def push(self, transition):
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        current_q_values = self.main_network(states).gather(1, actions)
        next_q_values = torch.max(self.target_network(next_states), dim=1)[0].reshape(-1, 1) * (1 - dones)
        target_q_values = rewards + self.gamma * next_q_values.detach()

        mse_loss = self.criterion(target_q_values, current_q_values)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % self.update_step:
            update_model(self.main_network, self.target_network, tau=self.tau)
            if self.eps > self.eps_threshold:
                self.eps *= self.eps_decay
            else:
                self.eps = self.eps_threshold

        return mse_loss.item()

    def write(self, reward, loss):
        self.writer.add_scalar('Loss/value_loss', loss, self.writer_step)
        self.writer.add_scalar('Reward', reward, self.writer_step)
        self.writer.add_scalar('Epsilon', self.eps, self.writer_step)
        self.writer_step += 1
    
if __name__ == "__main__":
        
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    num_episodes = 300

    agent = DQN(state_dim, action_dim)

    for episode in tqdm(range(num_episodes)):
        episode_reward = 0
        episode_loss = []
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            transition = [state, action, next_state, reward, done]
            agent.push(transition)
            state = next_state

            if agent.train_start():
                loss = agent.train()
                episode_loss.append(loss)

            if done:
                if agent.train_start():
                    agent.write(episode_reward, np.mean(episode_loss))
                break



