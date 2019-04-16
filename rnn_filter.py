import clevr_envs
import datetime
import gym
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from config import *
from derenderer import *
from sac import SAC
from tqdm import tqdm
from utils import *
from visualize import *

device = "cuda:3" if torch.cuda.is_available() else "cpu:0"

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s, a, o, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, o, d) 
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, o, d = map(torch.cat, zip(*batch))
        return s, a, o, d

    def __len__(self):
        return len(self.buffer)

class RNNFilter(nn.Module):
    def __init__(self):
        super(RNNFilter, self).__init__()

        self.h0 = nn.Parameter(torch.randn(b_num_layers, 1, dim_hidden))
        self.rb = ReplayBuffer(10000)

        self.filter     = nn.GRU(16, dim_hidden, b_num_layers)
        self.derenderer = Derenderer()
        self.derenderer.load("ckpt/dr.pt")
        self.decoder    = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_state)
        )
        self.opt_f = torch.optim.Adam(
                list(self.filter.parameters())+list(self.decoder.parameters()), 
                lr=3e-4)

    def forward(self, o_t, d_t, a_tm1=None, h_tm1=None):
        """
        o_t  : observation, [1, C, H, W]
        d_t  : depth,       [1, 1, H, W]
        a_tm1: action,      [1, 1]
        h_tm1: hidden state
        """
        if h_tm1 is None:
            batch = o_t.shape[0]
            h_tm1 = self.h0.repeat(1, batch, 1)
            a_tm1 = torch.zeros(batch, 1).to(device)

        d, e = self.derenderer(o_t, d_t)
        x    = rotate_state2(d, a_tm1)
        x    = torch.cat((x, e.unsqueeze(-1)), -1).view(1, -1, 16)
        x, h = self.filter(x, h_tm1)
        x    = self.decoder(x).view(-1, n_obj, dim_obj)
        return x, h

    def e2e_loss_fn(self, s_, s):
        """
        s_   : [:, n_obj, dim_obj]
        s    : [:, n_obj, dim_obj]
        """
        loss  = F.mse_loss(s_, s)
        return loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def to_device(self, x):
        return x.to(device)

    def update_parameters(self):
        S, A, O, D = self.rb.sample(batch_size)
        S, A, O, D = map(self.to_device, [S, A, O, D])

        n_steps = S.shape[1]

        ### end-to-end loss
        self.opt_f.zero_grad()
        e2e_loss = 0
        x, h = None, None
        for _ in range(n_steps):
            if _ == 0:
                x, h = self.forward(O[:,_,:,:,:], D[:,_,:,:,:])
            else:
                x, h = self.forward(O[:,_,:,:,:], D[:,_,:,:,:], A[:,_-1,:], h)
            e2e_loss += self.e2e_loss_fn(x, S[:,_,:,:])

        e2e_loss /= n_steps
        e2e_loss.backward()
        self.opt_f.step()
        
        return e2e_loss.item()

if __name__ == "__main__":
    env = gym.make('ActivePerception-v0')

    rnn = RNNFilter().to(device)

    opt  = optim.Adam(rnn.parameters(), lr=1e-4)

    # set up the experiment folder
    experiment_id = "rnn_" + get_datetime()
    save_path = CKPT+experiment_id+".pt"

    max_frames = 1000000
    frame_idx  = 0
    best_loss  = np.inf
    pbar       = tqdm(total=max_frames)
    while frame_idx < max_frames:
        pbar.update(1)

        S, A, O, D = [], [], [], []
        scene_data, obs = env.reset(False)

        s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
        o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
        d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth
        S.append(s); O.append(o); D.append(d)

        for step in range(1, 8):
            frame_idx += 1
            th  = np.pi/4*step
            obs = env.step(th)

            s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
            o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
            d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth
            a = torch.FloatTensor(1,1).fill_(th) # [1, 1]              action
            S.append(s); O.append(o); D.append(d); A.append(a)

        S = torch.cat(S).unsqueeze(0) # [1, 8, n_obj, dim_obj]
        A = torch.cat(A).unsqueeze(0) # [1, 8, 1]
        O = torch.cat(O).unsqueeze(0) # [1, 8, C, H, W]
        D = torch.cat(D).unsqueeze(0) # [1, 8, 1, H, W]
        rnn.rb.push(S, A, O, D)

        if len(rnn.rb) > batch_size:
            loss = rnn.update_parameters()
            if loss < best_loss:
                tqdm.write("[INFO] best loss %10.4f" % loss)
                best_loss = loss
                rnn.save_model(save_path)
    pbar.close()
