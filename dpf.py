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
from rnn_filter import *
from sac import SAC
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from utils import *
from visualize import *


class DPF(nn.Module):
    def __init__(self):
        super(DPF, self).__init__()

        self.rb = ReplayBuffer(9900)
        self.derenderer = Derenderer()
        self.derenderer.load("ckpt/dr.pt")

        ### the particle proposer
        self.generator = nn.Sequential(
            nn.Linear(16, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, K*dim_state)
        )

        ### the observation model (discriminator)
        self.discriminator = nn.Sequential(
            nn.Linear(dim_state+16, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )
        self.alpha = 0.8 
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=3e-4)
        self.opt_f = torch.optim.Adam(
                list(self.discriminator.parameters()) \
                        +list(self.generator.parameters()), 
                lr=3e-4)

    def forward(self, o_t, d_t, a_tm1=None, p_tm1=None, w_tm1=None, n_new=0):
        """
        params
            o_t  : [:, C, H, W]
            d_t  : [:, 1, H, W]
            a_tm1: [:, 1]
            p_tm1: [:, K, n_obj, dim_obj]
            w_tm1: [:, K]
        -------------------------
        return
            p_t  : [:, K, n_obj, dim_obj]
            w_t  : [:, K]
            p_n  : [B, K, dim_state]
        """
        B    = o_t.shape[0]
        if p_tm1 is None:
            a_tm1 = torch.zeros(B, 1).to(device)

        d, e = self.derenderer(o_t, d_t)
        x    = rotate_state2(d, a_tm1)
        x    = torch.cat((x, e.unsqueeze(-1)), -1).view(B, 16)
        p_n  = self.generator(x).view(B, K, n_obj, dim_obj)

        if p_tm1 is None:
            w_t = torch.Tensor(B, K).fill_(-np.log(K)).to(device)
            return p_n, w_t, p_n, x
        else:
            w_t = w_tm1 + self.update_belief(p_tm1, x)
            p_t = p_tm1
            if n_new > 0:
                p_t, w_t = self.resample(p_t, w_t, K-n_new)
                w_t = torch.cat((w_t, torch.Tensor(B, n_new).fill_(-np.log(K)).to(device)), 1)
                p_t = torch.cat((p_t, p_n[:,:n_new]), 1)
            elif n_new < 0: # just resample
                p_t, w_t = self.resample(p_t, w_t, K)
            w_t -= w_t.max(1, keepdim=True)[0]

        return p_t, w_t, p_n, x

    def update_belief(self, p, x):
        """
        p: [:, K, n_obj, dim_obj]
        x: [:, 16]
        """
        x = torch.cat((
            p.view(-1, K, dim_state), 
            x.unsqueeze(1).repeat(1,K,1)), -1) # [:, K, 16+dim_state]
        return self.discriminator(x).squeeze(-1)

    def resample(self, p_t, w_t, n):
        """
        soft-resampling
        """
        batch_size = p_t.shape[0]
        w_t = w_t - torch.logsumexp(w_t, 1, keepdim=True) # normalize

        # create uniform weights for soft resampling
        uniform_w = torch.Tensor(batch_size, K)
        uniform_w = uniform_w.fill_(-np.log(K)).to(device)

        if self.alpha < 1.0:
            q_w = torch.stack(
                [w_t+np.log(self.alpha), uniform_w+np.log(1-self.alpha)],
                -1
            ) # [:, n_particles, 2]

            q_w = torch.logsumexp(q_w, -1) # q_w=log(a*~w + (1-a)*1/n_particles)
            q_w = q_w - torch.logsumexp(q_w, -1, keepdim=True)
            w_t = w_t - q_w
        else:
            q_w = w_t
            w_t = uniform_w

        m = Categorical(logits=q_w) # sample n_old from old particles
        i = m.sample((n,)).t()
        w = w_t.gather(1, i)
        p = p_t.gather(1, i.view(-1,n,1,1).repeat(1,1,n_obj,dim_obj))
        return p, w

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def to_device(self, x):
        return x.to(device)

    def e2e_loss_fn(self, p, w, s):
        """
        calculate the negative log likelihood of s given (p, w)
        p: [B, K, n_obj, dim_obj]
        w: [B, K]
        s: [B, n_obj, dim_obj]
        """
        '''
        w = w - torch.logsumexp(w.detach(), -1, keepdim=True) # normalize w
        x = (p - s.unsqueeze(1)).pow(2).sum(-1).sum(-1) * 0.1 / 2
        # w: [B, K], x: [B, K]
        loss = -torch.logsumexp(w-x, 1).mean()
        '''
        w = F.softmax(w, -1).unsqueeze(2).unsqueeze(3)
        loss = F.mse_loss((p*w).sum(1), s)
        return loss

    def update_parameters(self):
        S, A, O, D = self.rb.sample(batch_size)
        S, A, O, D = map(self.to_device, [S, A, O, D])

        n_steps = S.shape[1]
        B       = S.shape[0]

        ### end-to-end loss
        self.opt_f.zero_grad()
        e2e_loss = 0
        d_loss = 0
        p, w = None, None
        for _ in range(n_steps):
            n_new = int(K* (0.7**_))
            if _ == 0:
                p, w, p_n, x = self.forward(
                        O[:,_,:,:,:], 
                        D[:,_,:,:,:], 
                        n_new=n_new)
            else:
                p, w, p_n, x = self.forward(
                        O[:,_,:,:,:], 
                        D[:,_,:,:,:], 
                        A[:,_-1,:], 
                        p, 
                        w, 
                        n_new)
            e2e_loss += self.e2e_loss_fn(p, w, S[:,_,:,:])
            sample = torch.cat(
                (S[:,_,:,:], p_n[:,np.random.randint(K),:,:].detach())
            ).view(-1, dim_state).detach() # [2B, dim_state]
            target  = torch.zeros(B*2, 1).to(device)
            target[:B,:] = 1
            scores  = self.discriminator(torch.cat((sample, x.detach().repeat(2,1)), -1))
            d_loss += F.binary_cross_entropy_with_logits(scores, target)

        e2e_loss /= n_steps
        d_loss /= n_steps
        (e2e_loss+d_loss*1e-1).backward()
        self.opt_f.step()
        '''

        ### adversarial loss
        p, w = None, None
        B = S.shape[0]
        samples, targets, X = [], [], []
        self.opt_g.zero_grad()
        self.discriminator.eval()
        g_loss = 0
        for _ in range(n_steps):
            n_new = int(K* (0.7**_))

            if _ == 0:
                p, w, p_n, x = self.forward(
                        O[:,_,:,:,:], 
                        D[:,_,:,:,:], 
                        n_new=n_new)
            else:
                p, w, p_n, x = self.forward(
                        O[:,_,:,:,:], 
                        D[:,_,:,:,:], 
                        A[:,_-1,:], 
                        p, 
                        w, 
                        n_new)
                samples.append(
                        torch.cat(
                            (S[:,_,:,:], p[:,np.random.randint(K),:,:].detach())
                        ).view(-1, dim_state)) # [2B, dim_state]
                target  = torch.zeros(B*2, 1).to(device)
                target[:B,:] = 1
                targets.append(target)
                X.append(x.detach())

                scores = self.discriminator(
                        torch.cat((
                            p_n.view(-1, K, dim_state), 
                            x.unsqueeze(1).repeat(1,K,1)
                        ), -1))
                target = torch.ones(B,K,1).to(device)
                g_loss += F.binary_cross_entropy_with_logits(scores, target) 
    
        (g_loss/n_steps).backward()
        self.opt_g.step()

        self.discriminator.train()
        d_loss = 0
        self.opt_d.zero_grad()
        for (sample, target, x) in zip(samples, targets, X):
            scores  = self.discriminator(torch.cat((sample, x.repeat(2,1)), -1))
            d_loss += F.binary_cross_entropy_with_logits(scores, target)
        (d_loss/n_steps).backward()
        self.opt_d.step()

        return g_loss.item()/n_steps, d_loss.item()/n_steps 
        '''
        return e2e_loss.item(), d_loss.item()

def train_dpf():
    env = gym.make('ActivePerception-v0')
    dpf = DPF().to(device)

    # set up the experiment folder
    experiment_id = "dpf_" + get_datetime()
    save_path = CKPT+experiment_id+".pt"

    max_frames = 1000000
    frame_idx  = 0
    best_loss  = np.inf
    pbar       = tqdm(total=max_frames)
    stats      = {
            'e2e_loss':[],
            'd_loss':[],
            }
    while frame_idx < max_frames:
        pbar.update(1)

        if frame_idx < 9900*8: # adding train set to buffer 
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
            dpf.rb.push(S, A, O, D)

        if len(dpf.rb) > batch_size:
            e, d = dpf.update_parameters()
            stats['e2e_loss'].append(e)
            stats['d_loss'].append(d)
            loss = e+d
            if loss < best_loss:
                tqdm.write("[INFO] best loss %10.4f (g: %10.4f, d:%10.4f)" % (loss,e,d))
                best_loss = loss 
                dpf.save_model(save_path)
            if frame_idx % 10 == 0:
                plot_training_f(stats, 'dpf', 'ckpt/dpf_train_curve.png')
    pbar.close()

def test_dpf(path, n_actions=1):
    env = gym.make('ActivePerception-v0')
    env.sid = 9900 # test
    dpf = DPF().to(device)
    dpf.load_model(path)

    mse = 0
    for episode in tqdm(range(100)):
        scene_data, obs = env.reset(False)

        s = get_state(scene_data).to(device) # [1, n_obj, dim_obj] state
        o = trans_rgb(obs['o']).to(device)   # [1, C, H, W]        rgb
        d = trans_d(obs['d']).to(device)     # [1, 1, H, W]        depth

        p, w, p_n, x = dpf(o, d, n_new=K)
        for step in range(n_actions):        # n_actions allowed
            th    = np.random.rand()*2*np.pi 
            obs   = env.step(th)
            o     = trans_rgb(obs['o']).to(device)
            d     = trans_d(obs['d']).to(device)
            n_new = int(K*(0.7**(step+1)))

            th    = torch.FloatTensor([th]).view(1, -1).to(device)
            p,w,p_n,x = dpf(o, d, th, p, w, n_new)
        mse += dpf.e2e_loss_fn(p,w,s).item()
    return mse

if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_dpf()
    else:
        print("[INFO] dpf mse: ", test_dpf(sys.argv[1]))
