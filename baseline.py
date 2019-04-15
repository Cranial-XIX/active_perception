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

        # batch size is 1
        self.h0 = nn.Parameter(torch.randn(b_num_layers, 1, dim_hidden))
        self.filter = nn.GRU(16, dim_hidden, b_num_layers)

        self.derenderer = Derenderer()

        self.decoder = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_state)
        )

    def forward(self, o_t, a_tm1=None, h_tm1=None):
        """
        o_t  : observation, [1, C, W, H]
        a_tm1: action,      [1, 1]
        h_tm1: hidden state
        """
        if h_tm1 is None:
            h_tm1 = self.h0
            a_tm1 = torch.zeros(1, 1).to(device)

        d, e = self.derenderer(o_t)
        x    = rotate_state(d, a_tm1, device)
        x    = torch.cat((x, e.unsqueeze(-1)), -1).view(-1, 16)
        x, h = self.filter(x, h_tm1)
        x    = self.decoder(x).view(-1, n_obj, dim_obj)
        return x, h

    def loss(self, s_, s):
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

if __name__ == "__main__":
    env = gym.make('ActivePerception-v0')

    rnn = RNNFilter().to(device)

    if len(sys.argv) > 1:
        print("pass")
    else:
        opt  = optim.Adam(rnn.parameters(), lr=1e-4)

        # set up the experiment folder
        experiment_id = "rnn_" + get_datetime()
        save_path = CKPT+experiment_id

        max_frames = 1000000
        frame_idx  = 0
        rewards    = []
        batch_size = 64

        pbar       = tqdm(total=max_frames)
        episode    = 0
        L          = 0
        while frame_idx < max_frames:
            episode += 1
            pbar.update(1)

            scene_data, obs = env.reset()
            target = get_state(scene_data).to(device) # [1, n_obj, dim_obj]

            rgb  = trans_rgb(obs['o']).to(device)      # [1, 3, 64, 64]
            x, h = rnn(rgb)
            loss = rnn.loss(x, target)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            L += loss.item()
            opt.step()

            for step in range(1, 8):
                a = np.pi/4*step
                obs = env.step(a)

                rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
                x, h = rnn(
                        rgb,
                        torch.FloatTensor([a]).view(1,1).to(device),
                        h)

                loss = rnn.loss(x, target)

                opt.zero_grad()
                loss.backward(retain_graph=True)
                L += loss.item()
                opt.step()
                frame_idx += 1

            if episode % 10 == 0:
                tqdm.write("[INFO] epi %05d | loss %10.4f |" % (episode, L/10))
                rnn.save_model(save_path+"_model.pt")
        pbar.close()
