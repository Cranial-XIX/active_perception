import datetime
import gym
import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import deepmind_lab

from tqdm import tqdm
from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s_t, a_t, r_t, s_tp1, done=False):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s_t, a_t, r_t, s_tp1, done)
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    def __init__(self, dim_state, dim_hidden, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(dim_state, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.cat((state), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, dim_state, n_actions, dim_hidden, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(dim_state + n_actions, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.linear3 = nn.Linear(dim_hidden, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self,
                dim_state,
                n_actions,
                dim_hidden,
                init_w=3e-3,
                log_std_min=-20,
                log_std_max=2):

        super(PolicyNetwork, self).__init__()

        self.device = "cuda:0"
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(dim_state, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)

        self.mean_linear = nn.Linear(dim_hidden, n_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(dim_hidden, n_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x       = F.relu(self.linear1(x))
        x       = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, x, epsilon=1e-6):
        mean, log_std = self.forward(x)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = 0
        high = np.pi*2

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high).astype(int)

        return action

    def _reverse_action(self, action):
        low  = 0
        high = np.pi*2

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high).astype(int)

        return actions

class SAC:
    def __init__(self):
        dim_action = 1
        dim_state  = 200
        dim_hidden = 256
        self.device = "cuda:0"

        self.value_net = ValueNetwork(dim_state, dim_hidden).to(self.device)
        self.target_value_net = ValueNetwork(dim_state, dim_hidden).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.soft_q_net = SoftQNetwork(dim_state, dim_action, dim_hidden).to(self.device)
        self.policy_net = PolicyNetwork(dim_state, dim_action, dim_hidden).to(self.device)

        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def save_model(self, path):
        stats = {}
        stats['v_net'] = self.value_net.state_dict()
        stats['q_net'] = self.soft_q_net.state_dict()
        stats['pi_net'] = self.policy_net.state_dict()
        stats['v_opt'] = self.value_optimizer.state_dict()
        stats['q_opt'] = self.soft_q_optimizer.state_dict()
        stats['pi_opt'] = self.policy_optimizer.state_dict()
        torch.save(stats, path)

    def load_model(self, path):
        stats = torch.load(path)
        self.value_net.load_state_dict(stats['v_net'])
        self.soft_q_net.load_state_dict(stats['q_net'])
        self.policy_net.load_state_dict(stats['pi_net'])
        self.value_optimizer.load_state_dict(stats['v_opt'])
        self.soft_q_optimizer.load_state_dict(stats['q_opt'])
        self.policy_optimizer.load_state_dict(stats['pi_opt'])

    def soft_q_update(self,
                    batch_size,
                    gamma=0.99,
                    mean_lambda=1e-3,
                    std_lambda=1e-3,
                    z_lambda=0.0,
                    soft_tau=1e-2,
                ):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
