import clevr_envs
import cv2
import gym
import torch
import torch.nn
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

class Observation(nn.Module):
    def __init__(self):

        super(Observation, self).__init__()
        self.derenderer = nn.Sequential(
            nn.Conv2d( 6, 8, 3, padding=1),
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
            nn.Linear(dim_hidden, dim_obs)
        )

        self.masks = []
        for _ in MASK:
            self.masks.append(
                    torch.FloatTensor(_).view(1, -1, 1, 1).repeat(B, 1, H, W)/255)

        self.opt = torch.optim.Adam(self.parameters(), 3e-4)

    def forward(self, x):
        derendered = []
        for m in self.masks:
            mask = ((x - m).abs().sum(1, keepdim=True).squeeze(1) < 1e-1) # [B, 1, H, W]
            seg  = x * mask
            derendered.append(torch.cat((x, seg), 1)) # [B, 6, H, W]
        return torch.cat(derendered, -1)

    def loss(self, y_hat, y):
        """
        y_hat: [B, dim_state]
        y    : [B, dim_state]
        """
        y_hat = y_hat.view(-1, n_obj, dim_obj) 
        y     = y.view(-1, n_obj, dim_obj)
        iy, iy_hat = y[:,:,0].unsqueeze(-1), y_hat[:,:,0].unsqueeze(-1)
        jy, jy_hat = y[:,:,1:], y_hat[:,:,1:]
        exist_loss = F.smooth_l1_loss(iy_hat, iy)
        l2_loss    = F.mse_loss(iy*jy_hat, iy*jy)
        return exist_loss, l2_loss

    def train(self):
        train_data = torch.load("data/observation.train")
        length = len(train_data)
        for episode in tqdm(range(1, 1001)):
            idx = np.random.choice(length, length, False)
            L = 0 
            for i in range(length//B):
                o, s = map(torch.cat, zip(*train_data[i*B:(i+1)*B]))
                o = o.to(device) # [B, C, H, W]
                s = s.to(device) # [B, dim_state]
                s_ = self.forward(o)
                self.opt.zero_grad()
                loss = self.loss(s_, s)
                loss.backward()
                self.opt.step()
                L += loss.item()
            L /= (length//B)
            tqdm.write("[INFO] epi %05d | loss %10.4f" % (episode, L))

def generate_data(total=100):
    env = gym.make('ActivePerception-v0')
    data = []
    for _ in tqdm(range(total)):
        scene_data, obs = env.reset()
        state = get_state(scene_data)
        data.append((trans_rgb(obs['o']), state))
        for j in range(10):
            th = np.random.rand()*2*np.pi
            obs = env.step(-th)
            state = rotate_state(state, th)
            data.append((trans_rgb(obs['o']), state))
    threshold = int(total * 0.9)
    train = data[:threshold]
    val   = data[threshold:]
    torch.save(train, "data/observation.train")
    torch.save(val, "data/observation.val")

def test_rotation():
    a = torch.Tensor([0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0]).unsqueeze(0)
    print(rotate_state(a, np.pi/2))

def test_mask():
    env = gym.make('ActivePerception-v0')
    _, obs = env.reset()
    tmp = obs['o']
    obs = trans_rgb(obs['o']) # [1, 3, H, W]

    B, _, H, W = obs.shape

    m = torch.FloatTensor(MASK[0]).view(1, -1, 1, 1).repeat(B, 1, H, W)/255
    seg = (obs - m).abs().sum(1).squeeze() # [B, H, W]
    seg = (seg < 1e-1) 
    Image.fromarray(tmp).save("original.png")
    plt.imshow(seg, cmap='gray')
    plt.savefig("kk.png")

if __name__ == "__main__":
