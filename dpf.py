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

class DPF(nn.Module):
    def __init__(self):
        super(DPF, self).__init__()

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
            nn.Linear(dim_hidden, dim_input)
        )

        self.proposer = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, K*dim_state)
        )

        self.observation = nn.Sequential(
            nn.Linear(dim_state+dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

    def save_model(self, path):
        stats = {}
        stats['e'] = self.encoder.state_dict()
        stats['p'] = self.proposer.state_dict()
        stats['o'] = self.observation.state_dict()
        torch.save(stats, path)

    def load_model(self, path):
        stats = torch.load(path)
        self.encoder.load_state_dict(stats['e'])
        self.proposer.load_state_dict(stats['p'])
        self.observation.load_state_dict(stats['o'])

    def forward(self, o_t, a_t=None, p_t=None, w_t=None):
        """
        o_t: [B, C, W, H]
        a_t: [B, 1]
        p_t: [B, K, dim_state]
        w_t: [B, K]
        -------------------------
        return
        p_t, w_t same shape
        proposed: [B, K, dim_state]
        """
        encoded_o_t = self.encoder(o_t) # [B, dim_input]
        B = o_t.shape[0]

        proposed = self.propose(encoded_o_t) # [B, K, dim_state]
        if p_t is None:
            w_t = torch.Tensor(B, K).fill_(-np.log(K)).to(device)
            #p_t = self.transit(proposed, a_t, cond)
            return proposed, w_t, proposed.mean(1)
        else:
            p_t = self.transit(p_t, a_t)
            w_t += self.update_weight(p_t, encoded_o_t)

        return p_t, w_t, proposed.mean(1)

    def propose(self, encoded_o_t):
        """
        o_t: [B, dim_input]
        propose particles [B, K, dim_state]
        """
        p = self.proposer(encoded_o_t).view(1, K, -1)
        
        e   = torch.sigmoid(p[:,:,0::4]).unsqueeze(-1)
        r   = F.relu(p[:,:,1::4]).unsqueeze(-1)
        phi = (torch.sigmoid(p[:,:,2::4]) * 2*np.pi).unsqueeze(-1)
        th  = (torch.sigmoid(p[:,:,3::4]) * np.pi).unsqueeze(-1)
        p   = torch.cat((e,r,phi,th), -1).view(-1, K, dim_state)
        return p

    def transit(self, s_t, a_t):
        """
        s_t:  [B, K, dim_state]
        a_t:  [B, 1]
        """
        s_t[:,:,2::4] += a_t.unsqueeze(1).repeat(1, K, 1)
        return s_t.fmod(2*np.pi)

    def update_weight(self, s_t, encoded_o_t):
        """
        s_t: [B, K, dim_state]
        o_t: [B, dim_input]
        """
        x = torch.cat((s_t, encoded_o_t.unsqueeze(0).repeat(1,K,1)), -1)
        return self.observation(x).squeeze(-1)

    def resample(self, s_t, w_t):
        """
        soft-resampling
        """
        batch_size, n_particles, dim_state = s_t.shape
        w_t = w_t - torch.logsumexp(w_t, 1, keepdim=True) # normalize

        # create uniform weights for soft resampling
        uniform_w = torch.Tensor(batch_size, n_particles)
        uniform_w = uniform_w.fill_(-np.log(n_particles)).to(self.cfg.device)

        if self.alpha < 1.0:
            q_w = torch.stack(
                [w_t+np.log(self.alpha), uniform_w+np.log(1-self.alpha)],
                -1
            ) # [:, n_particles, 2]

            # q_w = log(alpha*~w + (1-alpha)*1/n_particles)
            q_w = torch.logsumexp(q_w, -1)
            q_w = q_w - torch.logsumexp(q_w, -1, keepdim=True)
            w_t = w_t - q_w
        else:
            q_w = w_t
            w_t = uniform_w

        #n_new = int(n_particles * self.beta)
        n = n_particles

        m = Categorical(logits=q_w)
        idx = m.sample((n,)).t()    # [:, n]

        w_t = w_t.gather(1, idx)    # [:, n]

        # s_t [:, n_old, dim_state]
        s_t = s_t.gather(1, idx.unsqueeze(2).expand(-1, -1, dim_state))
        return s_t, w_t

if __name__ == "__main__":
    env = gym.make('ActivePerception-v0')

    sac = SAC()
    dpf = DPF().to(device)
    if len(sys.argv) > 1:
        if not os.path.exists(sys.argv[1]+"_model.pt"):
            print("[ERROR] Unknown path to load model")
            sys.exit(1)
        else:
            sac.load_model(sys.argv[1]+"_sac.pt")
            lstm.load_model(sys.argv[1]+"_model.pt")
        dpf.eval()
        max_steps  = 5

        scene_data, obs = env.reset()
        target = get_spherical_state(scene_data).reshape(-1).numpy()
    else:
        opt  = optim.Adam(dpf.parameters(), lr=1e-4)

        # set up the experiment folder
        experiment_id = "dpf_" + get_datetime()
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
            rgb = trans_rgb(obs['o']).to(device) # [1, 3, 64, 64]
            g, (h, c) = dpf(rgb)
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
