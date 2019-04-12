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

class Observation(nn.Module):
    def __init__(self):

        super(Observation, self).__init__()
        self.derenderer = nn.Sequential(
            nn.Conv2d( 3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d( 4, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            flatten(),
            nn.Linear(32*32*32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dim_obj)
        )

        self.masks = []
        for _ in MASK:
            self.masks.append(
                    torch.FloatTensor(_).view(1, -1, 1, 1).repeat(batch_size, 1, H, W).to(device)/255)

        self.opt = torch.optim.Adam(self.parameters(), 3e-4, weight_decay=1e-3)

    def forward(self, x):
        derendered = []
        for m in self.masks:
            mask = ((x - m).abs().sum(1, keepdim=True) < 1e-1) # [batch_size, 1, H, W]
            seg  = x * mask.float()
            dr   = self.derenderer(seg)
            derendered.append(dr)
        return torch.cat(derendered, -1)

    def loss(self, y_hat, y):
        """
        y_hat: [batch_size, dim_state]
        y    : [batch_size, dim_state]
        """
        y_hat = y_hat.view(-1, n_obj, dim_obj) 
        y     = y.view(-1, n_obj, dim_obj)
        iy, iy_hat = y[:,:,0].unsqueeze(-1), y_hat[:,:,0].unsqueeze(-1)
        jy, jy_hat = y[:,:,1:]*10, y_hat[:,:,1:]*10
        exist_loss = F.smooth_l1_loss(iy_hat, iy)
        l2_loss    = F.smooth_l1_loss(iy*jy_hat, iy*jy)
        return exist_loss, l2_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def pretrain(self):
        self.to(device)
        train_data = torch.load("data/observation.train")
        length = len(train_data)
        best_loss = np.inf
        for episode in tqdm(range(1, 1001)):
            idx = np.random.choice(length, length, False)
            E = L2 = 0 
            for i in range(length//batch_size):
                o, s = map(torch.cat, zip(*train_data[i*batch_size:(i+1)*batch_size]))
                o = o.to(device) # [batch_size, C, H, W]
                s = s.to(device) # [batch_size, dim_state]
                s_ = self.forward(o)
                self.opt.zero_grad()
                e, l2 = self.loss(s_, s)
                (e+l2).backward()
                self.opt.step()
                E += e.item()
                L2 += l2.item()
            L2 /= (length//batch_size)
            E /= (length//batch_size)
            if (episode % 5 == 0):
                tqdm.write("[INFO] epi %05d | l2 loss: %10.4f, e loss %10.4f" % (episode, L2, E))
            if (episode % 20 == 0):
                loss = self.test()
                self.train()
                if loss < best_loss:
                   best_loss = loss
                   self.save("ckpt/obs.pt")

    def test(self):
        self.eval()
        test_data = torch.load("data/observation.val")
        test_data = test_data[:128]
        length = len(test_data)
        best_loss = np.inf

        idx = np.random.choice(length, length, False)
        E = L2 = 0 
        for i in range(length//batch_size):
            o, s = map(torch.cat, zip(*test_data[i*batch_size:(i+1)*batch_size]))
            o = o.to(device) # [batch_size, C, H, W]
            s = s.to(device) # [batch_size, dim_state]
            s_ = self.forward(o)
            e, l2 = self.loss(s_, s)
            E += e.item()
            L2 += l2.item()
        L2 /= (length//batch_size)
        E /= (length//batch_size)
        tqdm.write("[INFO][TEST] l2 loss: %10.4f, e loss %10.4f" % (L2, E))
        return L2 + E

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

        for j in range(9):
            th = np.random.rand()*2*np.pi
            obs = env.step(th)
            Image.fromarray(obs['a'][::-1]).save("img/obs/%f.png" % th)
            s_ = self.forward(trans_rgb(obs['o']).to(device))
            s_ = rotate_state(s_.detach().cpu(), th).numpy().reshape(-1)
            objects.extend(get_scene(s_)['objects'])
        env.close()
        print(objects)
        visualize_o(objects)

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

    batch_size, _, H, W = obs.shape

    m = torch.FloatTensor(MASK[0]).view(1, -1, 1, 1).repeat(batch_size, 1, H, W)/255
    seg = (obs - m).abs().sum(1).squeeze() # [batch_size, H, W]
    seg = (seg < 1e-1) 
    Image.fromarray(tmp).save("original.png")
    plt.imshow(seg, cmap='gray')
    plt.savefig("kk.png")

if __name__ == "__main__":
   #generate_data(10000)
   obs = Observation() 
   #obs.pretrain()
   obs.visualize()
