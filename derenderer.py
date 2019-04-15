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
from visualize import *

device = "cuda:3" if torch.cuda.is_available() else "cpu:0"

class Derenderer(nn.Module):
    def __init__(self):
        super(Derenderer, self).__init__()

        self.derenderer = nn.Sequential(
            nn.Conv2d( 1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d( 8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            flatten(),
            nn.Linear(16*16*16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, dim_obj)
        )

        self.masks = []
        for _ in MASK:
            mask = torch.FloatTensor(_).view(1,-1,1,1)
            mask = mask.repeat(batch_size,1,H,W).to(device)/255-0.5
            self.masks.append(mask)

        self.opt = torch.optim.Adam(self.parameters(), 3e-4, weight_decay=1e-3)

    def forward(self, x):
        derendered = []
        exists = []
        for m in self.masks:
            mask  = ((x - m).abs().sum(1, keepdim=True) < 1e-1) # [:, 1, H, W]
            dr    = self.derenderer(mask.float()).unsqueeze(1)  # [:, 1, dim_obj]
            exist = (mask.view(batch_size, -1).sum(1, keepdim=True) > 0).float()
            derendered.append(dr)
            exists.append(exist)
        return torch.cat(derendered, 1), torch.cat(exists, -1)

    def loss(self, s_, s, exist):
        """
        s_   : [:, n_obj, dim_obj]
        s    : [:, n_obj, dim_obj]
        exist: [:, n_obj]
        """
        exist = exist.unsqueeze(-1)
        loss  = F.mse_loss(exist*s_, exist*s)
        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def pretrain(self):
        self.to(device)
        train_data = torch.load("data/observation.train")
        test_data  = torch.load("data/observation.val")
        o_te, s_te = map(torch.cat, zip(*test_data))
        o_tr, s_tr = map(torch.cat, zip(*train_data))
        length = len(train_data)
        del train_data; del test_data

        best_loss = np.inf
        for episode in tqdm(range(1, 1001)):
            L   = 0
            idx = np.random.choice(length, length, False)
            observations, states = o_tr[idx], s_tr[idx]
            for _ in range(0, length-batch_size, batch_size):
                o     = o_tr[_:_+batch_size].to(device)
                s     = s_tr[_:_+batch_size].to(device)
                s_, e = self.forward(o)

                self.opt.zero_grad()
                loss  = self.loss(s_, s, e)
                loss.backward()
                self.opt.step()
                L    += loss.item()
            L /= (length//batch_size)
            if (episode % 10 == 0):
                tqdm.write("[INFO] epi %05d | loss: %10.4f |" % (episode, L))
            if (episode % 20 == 0):
                loss = self.test(o_te, s_te)
                if loss < best_loss:
                    best_loss = loss
                    tqdm.write("[INFO] best val loss: %10.4f" % best_loss)
                    self.save("ckpt/obs.pt")

    def test(self, o_te, s_te):
        self.eval()
        length = o_te.shape[0] 
        L   = 0
        idx = np.random.choice(length, length, False)
        observations, states = o_te[idx], s_te[idx]
        for _ in range(0, length-batch_size, batch_size):
            o     = o_te[_:_+batch_size].to(device)
            s     = s_te[_:_+batch_size].to(device)
            s_, e = self.forward(o)
            L    += self.loss(s_, s, e).item()
        L /= (length//batch_size)
        self.train()
        return L

    def visualize(self):
        self.load("ckpt/obs.pt")
        self.to(device)
        self.eval()
        objects = []
        env = gym.make('ActivePerception-v0')
        check_path("img/obs")
        scene_data, obs = env.reset()
        Image.fromarray(obs['a'][::-1]).save("img/obs/0.png")
        s_ = self.forward(trans_rgb(obs['o']).to(device)).detach().cpu().numpy().reshape(-1)
        objects.extend(get_scene(s_)['objects'])

        for j in range(2):
            th = np.random.rand()*2*np.pi
            obs = env.step(th)
            Image.fromarray(obs['a'][::-1]).save("img/obs/%f.png" % th)
            s_ = self.forward(trans_rgb(obs['o']).to(device))
            s_ = rotate_state(s_.detach().cpu(), th).numpy().reshape(-1)
            objects.extend(get_scene(s_)['objects'])
        env.close()
        visualize_o(objects)

def generate_data(total=100):
    env = gym.make('ActivePerception-v0')
    data = []
    for _ in tqdm(range(total)):
        scene_data, obs = env.reset()
        state = get_state(scene_data)
        data.append((trans_rgb(obs['o']), state))
        for j in range(1, 8):
            th = np.pi/4*j 
            obs = env.step(th)
            state = rotate_state(state, -th)
            data.append((trans_rgb(obs['o']), state))
    threshold = int(total * 0.99)
    train = data[:threshold]
    val   = data[threshold:]
    torch.save(train, "data/observation.train")
    torch.save(val, "data/observation.val")

def test_rotation():
    a = torch.Tensor([
        [1,0,0],
        [0,1,0],
        [0,0,0],
        [0,0,0]]).unsqueeze(0)
    s = rotate_state(a, np.pi/2) # counter clockwise rotate 90 degree
    assert (s - torch.Tensor([
        [0,1,0],
        [-1,0,0],
        [0,0,0],
        [0,0,0]]).unsqueeze(0)).abs().sum() < 1e-10

def test_mask():
    env = gym.make('ActivePerception-v0')
    _, obs = env.reset()
    tmp = obs['o']
    obs = trans_rgb(obs['o']) # [1, 3, H, W]

    B, _, H, W = obs.shape

    seg = []
    for m in MASK:
        mask = torch.FloatTensor(m).view(1,-1,1,1).repeat(B, 1, H, W)/255-0.5
        seg.append((obs - mask).abs().sum(1).squeeze() < 1e-1)
    seg = seg[0] + seg[1] + seg[2] + seg[3]
    Image.fromarray(tmp.squeeze()).save("test_mask_rgb.png")
    plt.imshow(seg, cmap='gray')
    plt.savefig("test_mask_mask.png")

if __name__ == "__main__":
    #test_mask()
    #test_rotation()
    generate_data(10000)
    dr = Derenderer() 
    dr.pretrain()
    #dr.visualize()
