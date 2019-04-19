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

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.derenderer = Derenderer()
        self.derenderer.load("ckpt/dr.pt")

    def forward(self, o_t, d_t, a_tm1=None):
        if a_tm1 is None:
            batch = o_t.shape[0]
            a_tm1 = torch.zeros(batch, 1).to(device)

        d, e = self.derenderer(o_t, d_t)
        x    = rotate_state2(d, a_tm1)
        e    = e.unsqueeze(-1)
        x    = x*e
        return x, e

class RNNFilter(nn.Module):
    def __init__(self):
        super(RNNFilter, self).__init__()

        self.h0 = nn.Parameter(torch.randn(b_num_layers, 1, dim_hidden))
        self.rb = ReplayBuffer(9900)

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

def train_filter():
    env = gym.make('ActivePerception-v0')
    rnn = RNNFilter().to(device)

    # set up the experiment folder
    experiment_id = "rnn_" + get_datetime()
    save_path = CKPT+experiment_id+".pt"

    max_frames = 1000000
    frame_idx  = 0
    best_loss  = np.inf
    pbar       = tqdm(total=max_frames)
    stats      = {'losses': []} 
    while frame_idx < max_frames:
        pbar.update(1)

        if frame_idx < 9900*8: # adding train set to buffer 
            S, A, O, D = [], [], [], []
            scene_data, obs = env.reset(False)

            s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
            o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
            d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth
            S.append(s); O.append(o); D.append(d)

            steps = np.random.choice(7,7,False)+1
            for step in steps:
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
            stats['losses'].append(loss)
            if loss < best_loss:
                tqdm.write("[INFO] best loss %10.4f" % loss)
                best_loss = loss
                rnn.save_model(save_path)
            if frame_idx % 10 == 0:
                plot_training_f(stats, 'rnn', 'ckpt/rnn_filter_training_curve.png')
    pbar.close()

def test_baseline(n_actions=1, threshold=0.02):
    env = gym.make('ActivePerception-v0')
    env.sid = 9900 # test
    base = Baseline().to(device)

    mse = 0
    reward = 0
    for episode in tqdm(range(100)):
        scene_data, obs = env.reset(False)

        s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
        o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
        d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth

        x, e = base(o, d)
        s_   = x                             # [1, n_obj, dim_obj]
        e_   = e                             # [1, n_obj, 1]

        idx  = np.random.choice(7, 7, False)+1
        #for step in range(n_actions):        # n_actions allowed
        for step in idx:
            th   = step*np.pi/4#np.random.rand()*2*np.pi
            obs  = env.step(th)
            o = trans_rgb(obs['o']).to(device)
            d = trans_d(obs['d']).to(device)

            th   = torch.FloatTensor([th]).view(1, -1).to(device)
            x, e = base(o, d, th)
            s_  += x
            e_  += e
            err  = F.mse_loss(s_ / (e_+1e-10), s).item()
            d        = (err < threshold)
            r        = 8 if d else -1
            reward  += r
            if d:
                break

        s_ /= (e_+1e-10)
        mse += F.mse_loss(s_, s).item()#(s - s_).pow(2).sum().sqrt().item()
    print("baseline avg reward ", reward/100)
    return mse

def test_rnn(path, n_actions=1):
    env = gym.make('ActivePerception-v0')
    env.sid = 9900 # test
    rnn = RNNFilter().to(device)
    rnn.load_model(path)

    mse = 0
    for episode in tqdm(range(100)):
        scene_data, obs = env.reset(False)

        s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
        o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
        d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth

        s_, h = rnn(o, d)
        for step in range(n_actions):        # n_actions allowed
            th   = np.random.rand()*2*np.pi 
            obs  = env.step(th)
            o = trans_rgb(obs['o']).to(device)
            d = trans_d(obs['d']).to(device)

            th    = torch.FloatTensor([th]).view(1, -1).to(device)
            s_, h = rnn(o, d, th, h)
        mse += F.mse_loss(s_, s).item()#(s - s_).pow(2).sum().sqrt().item()
    return mse

def train_rnn_sac(path, threshold=0.02):
    env = gym.make('ActivePerception-v0')
    rnn = RNNFilter().to(device)
    rnn.load_model(path)
    sac = SAC()

    # set up the experiment folder
    experiment_id = "rsac_" + get_datetime()
    save_path = CKPT+experiment_id+".pt"

    max_frames = 100000
    frame_idx  = 0
    best_loss  = np.inf
    pbar       = tqdm(total=max_frames)
    stats      = {'losses': []} 
    best_reward = 0 
    avg_reward = 0
    avg_mse    = 0
    episode    = 0
    while frame_idx < max_frames:
        pbar.update(1)

        episode += 1
        env.sid = env.sid % 9900
        scene_data, obs = env.reset(False)

        S, A, R, D = [], [], [], []
        s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
        o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
        d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth

        s_, h = rnn(o, d)
        prev_mse = F.mse_loss(s_, s).item()
        h_numpy = h.view(-1).detach().cpu().numpy()
        S.append(h_numpy)
        for _ in range(8):
            frame_idx += 1
            th    = sac.policy_net.get_action(h_numpy.reshape(1,-1))
            obs   = env.step(th.item())
            o     = trans_rgb(obs['o']).to(device)
            d     = trans_d(obs['d']).to(device)
            th    = torch.FloatTensor([th]).view(1, -1).to(device)
            s_, h = rnn(o, d, th, h)

            mse      = F.mse_loss(s_, s).item()
            #r        = (mse - prev_mse)*100
            prev_mse = mse
            d        = (mse < threshold)
            r        = 8 if d else -1
            h_numpy  = h.view(-1).detach().cpu().numpy()

            S.append(h_numpy)
            A.append(th.cpu().numpy().reshape(-1))
            R.append(r)
            D.append(d)
            if d:
                break

        S, NS = S[:-1], S[1:]
        for s, a, r, ns, d in zip(S, A, R, NS, D):
            sac.replay_buffer.push(s, a, r, ns, d)
            if len(sac.replay_buffer) > batch_size:
                sac.soft_q_update(batch_size)

        avg_reward += np.array(r).sum()
        avg_mse    += prev_mse
        if episode % 10 == 0:
            avg_reward /= 10
            avg_mse /= 10
            tqdm.write("[INFO] epi %05d | avg r: %10.4f | avg mse: %10.4f" % (episode, avg_reward, avg_mse))
            if avg_reward > best_reward:
                best_reward = avg_reward
                sac.save_model(save_path)
            avg_reward = 0
            avg_mse = 0

def test_rnn_sac(r_path, s_path, threshold=0.02):
    env = gym.make('ActivePerception-v0')
    env.sid = 9900 # test
    rnn = RNNFilter().to(device)
    rnn.load_model(r_path)
    sac = SAC()
    sac.load_model(s_path)

    reward = 0
    for episode in tqdm(range(100)):
        scene_data, obs = env.reset(False)

        s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
        o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
        d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth

        s_, h = rnn(o, d)
        h_numpy = h.view(-1).detach().cpu().numpy()
        for step in range(8):        # n_actions allowed
            th    = sac.policy_net.get_action(h_numpy.reshape(1,-1)).item()
            obs  = env.step(th)
            o = trans_rgb(obs['o']).to(device)
            d = trans_d(obs['d']).to(device)
            th    = torch.FloatTensor([th]).view(1, -1).to(device)
            s_, h = rnn(o, d, th, h)
            h_numpy = h.view(-1).detach().cpu().numpy()
            mse      = F.mse_loss(s_, s).item()
            d        = (mse < threshold)
            r        = 8 if d else -1
            reward  += r
            if d:
                break

    print("avg reward ", reward/100)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_filter()
    elif sys.argv[1] == 's':
        train_rnn_sac(sys.argv[2])
    elif sys.argv[1] == 'st':
        test_rnn_sac(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'r':
        print("[INFO] rnn mse: ", test_rnn(sys.argv[2], int(sys.argv[3])))
    elif sys.argv[1] == 'b':
        print("[INFO] baseline mse: ",test_baseline(int(sys.argv[2])))
    else:
        print("[ERROR] Unknown flag")
