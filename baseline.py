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

class RNNFilter(nn.Module):
    def __init__(self):
        super(RNNFilter, self).__init__()

        # batch size is 1
        self.h0 = nn.Parameter(torch.randn(b_num_layers, 1, dim_hidden))
        self.filter = nn.GRU(dim_state, dim_hidden, b_num_layers)

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
            nn.Linear(64*64, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_state)
        )
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

        x          = self.encoder(o_t)
        x          = self.sphericalize(x)
        s_from_obs = self.rotate_back(x, a_tm1).view(1,x.shape[0],-1)

        x, h       = self.filter(s_from_obs, h_tm1)
        s_from_gru = self.sphericalize(self.decoder(x.view(1, -1)))
        return s_from_obs, s_from_gru, h

    def sphericalize(self, s):
        """
        transform the state into a spherical coordinates
        s     = (exist, r, phi, theta)
        exist ~ [0, 1]
        r     ~ [0, \inf)
        phi   ~ (0, 2\pi)
        th    ~ (0, \pi)
        """
        e   = torch.sigmoid(s[:,0::4]).unsqueeze(-1)
        r   = F.relu(s[:,1::4]).unsqueeze(-1)
        phi = (torch.sigmoid(s[:,2::4]) * 2*np.pi).unsqueeze(-1)
        th  = (torch.sigmoid(s[:,3::4]) * np.pi).unsqueeze(-1)
        s   = torch.cat((e, r, phi, th), -1).view(-1, dim_state)
        return s

    def rotate_back(self, s, a):
        """
        rotate the current state back to the original view point
        """
        s[:,2::4] += a
        return s.fmod(2*np.pi)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        return

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    env = gym.make('ActivePerception-v0')

    sac = SAC()
    rnn = RNNFilter().to(device)

    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]+"_model.pt"):
            print("[ERROR] Unknown path to load model")
            sys.exit(1)
        else:
            #sac.load_model(sys.argv[1]+"_sac.pt")
            rnn.load_model(sys.argv[1]+"_model.pt")

        rnn.eval()
        max_steps  = 3

        scene_data, obs = env.reset()
        target = get_spherical_state(scene_data).reshape(-1).numpy()
        print("[INFO] Target is ", target)

        O_pred, RNN_pred = [], []

        rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
        go, gr, h = rnn(rgb)
        #guess_state = h.view(h.shape[0], -1).detach().cpu()
        O_pred.append(go.detach().cpu().view(-1).numpy())
        RNN_pred.append(gr.detach().cpu().view(-1).numpy())

        for step in range(max_steps):
            # SAC planning
            #a = sac.policy_net.get_action(guess_state.reshape(1, -1))
            a = 2 * np.pi / (max_steps+1) * (step+1)
            obs = env.step(a)

            rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
            go, gr, h = rnn(
                    rgb,
                    torch.FloatTensor([a]).view(1,1).to(device),
                    h)
            O_pred.append(go.detach().cpu().view(-1).numpy())
            RNN_pred.append(gr.detach().cpu().view(-1).numpy())
            #guess_state = h.view(h.shape[0], -1).detach().cpu()

        visualize_b(target, RNN_pred)
        visualize_b(target, O_pred, False)
    else:
        opt  = optim.RMSprop(rnn.parameters(), lr=1e-4)

        # set up the experiment folder
        experiment_id = "rnn_" + get_datetime()
        save_path = CKPT+experiment_id

        max_frames = 1000000
        max_steps  = 3
        frame_idx  = 0
        rewards    = []
        batch_size = 64

        episode    = 0
        bptt_step  = 10

        # regression loss for filtering
        criterion = LossFn() 

        # Update, use HER to train for sparse reward
        best_reward = -1
        pbar = tqdm(total=max_frames)
        while frame_idx < max_frames:
            episode += 1
            pbar.update(1)

            scene_data, obs = env.reset()
            target = get_spherical_state(scene_data).to(device)

            #states, actions, rewards, dones = [], [], [], []
            loss_rnn_e = loss_rnn_l2 = loss_obs_e = loss_obs_l2 = 0
            rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
            go, gr, h = rnn(rgb)

            #guess_state = h.view(h.shape[0], -1).detach().cpu()
            #states.append(guess_state)

            lre, lrl = criterion(gr, target)
            loe, lol = criterion(go, target)
            loss_rnn_e  += lre
            loss_rnn_l2 += lrl
            loss_obs_e  += loe
            loss_obs_l2 += lol
            opt.zero_grad()
            (lre+lrl+loe+lol).backward(retain_graph=True)
            opt.step()

            for step in range(max_steps):
                # SAC planning
                #a = sac.policy_net.get_action(guess_state.reshape(1,-1))
                a = (2*np.pi)/(max_steps+1)*(step+1)
                obs = env.step(a)

                if (step+1) % bptt_step == 0:
                    h = h.detach()

                rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
                go, gr, h = rnn(
                        rgb,
                        torch.FloatTensor([a]).view(1,1).to(device),
                        h)
                #guess_state = h.view(h.shape[0], -1).detach().cpu()

                lre, lrl = criterion(gr, target)
                loe, lol = criterion(go, target)
                loss_rnn_e  += lre
                loss_rnn_l2 += lrl
                loss_obs_e  += loe
                loss_obs_l2 += lol
                opt.zero_grad()
                (lre+lrl+loe+lol).backward(retain_graph=True)
                opt.step()

                '''
                l      = loss.item()
                reward = np.exp(-l + 3)
                done   = l < 5e-3

                states.append(guess_state)
                actions.append(np.array([a.item()]))
                rewards.append(np.array([reward]))
                dones.append(done)
                '''

                frame_idx += 1
                '''
                if done:
                    tqdm.write("[INFO] solved in %d steps!" % (step+1))
                    break
                '''

            # Replay buffer
            '''
            states, nstates = states[:-1], states[1:]
            for i, (s, a, r, n, d) in enumerate(zip(states, actions, rewards, nstates, dones)):
                sac.replay_buffer.push(s, a, r, n, d)
            '''

            if episode % 10 == 0:
                tqdm.write("[INFO] episode %10d " \
                        "| loss_rnn_e %10.4f " \
                        "| loss_rnn_l2 %10.4f " \
                        "| loss_obs_e %10.4f " \
                        "| loss_obs_l2 %10.4f " % (
                            episode,
                            loss_rnn_e,
                            loss_rnn_l2,
                            loss_obs_e,
                            loss_obs_l2))
                rnn.save_model(save_path+"_model.pt")

        pbar.close()
