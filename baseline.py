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
from sac import SAC
from utils import *
from tqdm import tqdm
from visualize import *

device = "cuda:3" if torch.cuda.is_available() else "cpu:0"

class LossFn(nn.Module):
    def __init__(self):
        super(LossFn, self).__init__()
    
    def forward(self, y_hat, y):
        """
        y_hat: [B, dim-state]
        y    : [B, dim-state]
        """
        y_hat = y_hat.view(-1, n_obj, dim_obj) 
        y     = y.view(-1, n_obj, dim_obj)
        iy, iy_hat = y[:,:,0].unsqueeze(-1), y_hat[:,:,0].unsqueeze(-1)
        jy, jy_hat = y[:,:,1:], y_hat[:,:,1:]

        exist_loss = F.smooth_l1_loss(iy_hat, iy)
        l2_loss    = F.mse_loss(iy*jy_hat, iy*jy)

        return exist_loss, l2_loss

class LSTMFilter(nn.Module):
    def __init__(self):
        super(LSTMFilter, self).__init__()

        # batch size is 1
        self.h0 = nn.Parameter(torch.randn(b_num_layers, 1, b_dim_hidden))
        self.c0 = nn.Parameter(torch.randn(b_num_layers, 1, b_dim_hidden))
        self.filter = nn.LSTM(b_dim_input, b_dim_hidden, b_num_layers)

        self.encoder = nn.Sequential(
            nn.Conv2d( 3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d( 8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            flatten(),
            nn.Linear(64*64, b_dim_hidden),
            nn.ReLU(),
            nn.Linear(b_dim_hidden, b_dim_input)
        )
        self.decoder = nn.Sequential(
            nn.Linear(b_dim_hidden, b_dim_hidden),
            nn.ReLU(),
            nn.Linear(b_dim_hidden, dim_state)
        )

    def forward(self, o_t, a_t=None, h=None, c=None):
        """
        o_t: observation, [1, C, W, H]
        a_t: action,      [1, 1]
        h  : hidden state
        c  : cell state
        """
        if h is None:
            h = self.h0
            c = self.c0
            a_t = torch.zeros(1, 1).to(device)

        x = self.encoder(o_t).view(1, 1, -1) # [1, 1, b_dim_input]
        #x = torch.cat((x, a_t), -1).view(1, 1, -1)
        x, (h, c) = self.filter(x, (h, c))
        hat_s_t = self.decoder(x.view(1, -1))
        hat_s_t = self.transform(hat_s_t, a_t)
        return hat_s_t, (h, c)

    def transform(self, s_t, a_t):
        """
        transform the state into a spherical coordinates
        s = (exist, r, phi, theta)
        exist ~ [0, 1]
        r     ~ [0, \inf)
        phi   ~ (0, 2\pi)
        th    ~ (0, \pi)
        """
        e   = torch.sigmoid(s_t[:,0::4])
        r   = F.relu(s_t[:,1::4])
        phi = (torch.sigmoid(s_t[:,2::4]) * 2*np.pi - a_t).fmod(2*np.pi)
        th  = torch.sigmoid(s_t[:,3::4]) * np.pi
        '''
        x   = r * torch.sin(th) * torch.cos(phi)
        y   = r * torch.sin(th) * torch.sin(phi)
        z   = r * torch.cos(th)
        e, x, y, z = e.unsqueeze(-1), x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)
        '''
        e   = e.unsqueeze(-1)
        r   = r.unsqueeze(-1)
        phi = phi.unsqueeze(-1)
        th  = th.unsqueeze(-1)
        s_t = torch.cat((e, r, phi, th), -1).view(-1, dim_state)
        return s_t

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    env = gym.make('ActivePerception-v0')

    sac  = SAC()
    lstm = LSTMFilter().to(device)
    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]+"_model.pt"):
            print("[ERROR] Unknown path to load model")
            sys.exit(1)
        else:
            sac.load_model(sys.argv[1]+"_sac.pt")
            lstm.load_model(sys.argv[1]+"_model.pt")
        lstm.eval()
        max_steps  = 5

        scene_data, obs = env.reset()
        target = get_spherical_state(scene_data).reshape(-1).numpy()
        print("[INFO] Target is ", target)

        predictions = []

        rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
        g, (h, c) = lstm(rgb)
        B = h.shape[0]
        guess_state = torch.cat((h.view(B,-1), c.view(B,-1)), -1).detach().cpu()
        predictions.append(g.detach().cpu().view(-1).numpy())

        for step in range(max_steps):
            # SAC planning
            a = sac.policy_net.get_action(guess_state.reshape(1,-1))
            obs = env.step(a.item())

            rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
            g, (h, c) = lstm(
                    rgb,
                    torch.FloatTensor(a).view(1,1).to(device),
                    h, c)
            predictions.append(g.detach().cpu().view(-1).numpy())
            #guess_state = g.detach().cpu().squeeze().numpy()
            B = h.shape[0]
            guess_state = torch.cat((h.view(B,-1), c.view(B,-1)), -1).detach().cpu()

        visualize_b(target, predictions)
    else:
        opt  = optim.Adam(lstm.parameters(), lr=1e-4)

        # set up the experiment folder
        experiment_id = "lstm_" + get_datetime()
        save_path = CKPT+experiment_id
        img_path = IMG+experiment_id
        check_path(img_path)

        max_frames = 1000000
        max_steps  = 5
        frame_idx  = 0
        rewards    = []
        batch_size = 64

        episode    = 0
        bptt_step  = 10

        # regression loss for filtering
        criterion = LossFn() 

        # Update, use HER to train for sparse reward
        best_reward = -1
        pbar = tqdm(total=frame_idx)
        best_loss = np.inf
        while frame_idx < max_frames:
            episode += 1
            pbar.update(1)

            scene_data, obs = env.reset()
            target = get_spherical_state(scene_data).to(device)

            states, actions, rewards, dones = [], [], [], []
            episode_loss = 0
            episode_l2_loss = 0
            h, c, a = None, None, None
            rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
            g, (h, c) = lstm(rgb)
            #guess_state = g.detach().cpu().squeeze().numpy()
            B = h.shape[0]
            guess_state = torch.cat((h.view(B,-1), c.view(B,-1)), -1).detach().cpu()
            states.append(guess_state)
            e_loss, l2_loss = criterion(g, target)
            loss = e_loss + l2_loss
            opt.zero_grad(); loss.backward(retain_graph=True); opt.step()
            episode_loss += loss.item()
            episode_l2_loss += l2_loss.item()

            for step in range(max_steps):
                # SAC planning
                a = sac.policy_net.get_action(guess_state.reshape(1,-1))
                obs = env.step(a.item())

                if (step+1) % bptt_step == 0:
                    h, c = h.detach(), c.detach()

                rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
                g, (h, c) = lstm(
                        rgb,
                        torch.FloatTensor(a).view(1,1).to(device),
                        h, c)
                #guess_state = g.detach().cpu().squeeze().numpy()
                B = h.shape[0]
                guess_state = torch.cat((h.view(B,-1), c.view(B,-1)), -1).detach().cpu()
                e_loss, l2_loss = criterion(g, target)
                loss = e_loss + l2_loss
                opt.zero_grad(); loss.backward(retain_graph=True); opt.step()

                l = loss.item()
                episode_loss += l
                episode_l2_loss += l2_loss.item()
                reward = np.exp(-l + 3)
                done   = l < 5e-3

                states.append(guess_state)
                actions.append(np.array([a.item()]))
                rewards.append(np.array([reward]))
                dones.append(done)

                frame_idx += 1
                if done:
                    tqdm.write("[INFO] solved in %d steps!" % (step+1))
                    break

            # Replay buffer
            states, nstates = states[:-1], states[1:]
            for i, (s, a, r, n, d) in enumerate(zip(states, actions, rewards, nstates, dones)):
                sac.replay_buffer.push(s, a, r, n, d)

            if episode % 10 == 0:
                tqdm.write("[INFO] episode %10d | loss %10.4f "\
                        "| best %10.4f | l2_loss %10.4f" % (
                            episode, episode_loss, best_loss, episode_l2_loss))
                if episode_loss < best_loss:
                    best_loss = episode_loss
                    sac.save_model(save_path+"_sac.pt")
                    lstm.save_model(save_path+"_model.pt")

        pbar.close()
