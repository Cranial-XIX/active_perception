import clevr_envs
import cv2
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.nn.functional as F

from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import *
from visualize import *

device = "cuda:3" if torch.cuda.is_available() else "cpu:0"

class Derenderer(nn.Module):
    def __init__(self):
        super(Derenderer, self).__init__()

        '''
        self.derenderer = nn.Sequential(
            nn.Conv2d( 3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d( 4, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d( 8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d( 16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            flatten(),
            nn.Linear(16*32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dim_obj)
        )'''
        self.derenderer = nn.Sequential(
                nn.Conv2d(7,8,5,2,2),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.MaxPool2d(3,2,1),
                nn.Conv2d(8,8,3,padding=1),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.MaxPool2d(3,2,1),
                flatten(),
                nn.Linear(512, 64),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(64, dim_obj)
        )

        self.masks = []
        for _ in MASK:
            mask = torch.FloatTensor(_).view(1,-1,1,1)
            mask = mask.repeat(batch_size,1,H,W).to(device)/255-0.5
            self.masks.append(mask)

        self.opt = torch.optim.Adam(self.parameters(), 2e-4, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min')

    def forward(self, o, d):
        derendered = []
        exists = []
        for m in self.masks:
            mask  = ((o - m).abs().sum(1, keepdim=True) < 1e-1).float() # [:, 1, H, W]
            x     = torch.cat((o, mask*o,mask*d), 1)
            dr    = self.derenderer(x).unsqueeze(1)*2                   # [:, 4, dim_obj]
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
        exist = exist
        loss  = [F.mse_loss(exist*s_[:,:,_], exist*s[:,:,_]) for _ in range(3)]
        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def pretrain(self):
        self.to(device)
        writer = SummaryWriter()
        train_data = torch.load("data/observation.train")
        test_data  = torch.load("data/observation.val")
        o_te, d_te, s_te = map(torch.cat, zip(*test_data))
        o_tr, d_tr, s_tr = map(torch.cat, zip(*train_data))
        length = len(train_data)
        del train_data; del test_data

        best_loss = np.inf
        for episode in tqdm(range(1, 1001)):
            Lx = Ly = Lz = 0
            idx = np.random.choice(length, length, False)
            o_tr, d_tr, s_tr = o_tr[idx], d_tr[idx], s_tr[idx]
            for _ in range(0, length-batch_size, batch_size):
                o     = o_tr[_:_+batch_size].to(device)
                d     = d_tr[_:_+batch_size].to(device)
                s     = s_tr[_:_+batch_size].to(device)
                s_, e = self.forward(o, d)

                self.opt.zero_grad()
                lx, ly, lz = self.loss(s_, s, e)
                (lx+ly+lz).backward()
                self.opt.step()
                Lx   += lx.item()
                Ly   += ly.item()
                Lz   += lz.item()
            divide = length // batch_size
            Lx /= divide; Ly /= divide; Lz /= divide
            writer.add_scalar('tr_x', Lx, episode)
            writer.add_scalar('tr_y', Ly, episode)
            writer.add_scalar('tr_z', Lz, episode)

            xx, yy, zz = self.test(o_te, d_te, s_te)
            self.scheduler.step(xx+yy+zz)
            writer.add_scalar('te_x', xx, episode)
            writer.add_scalar('te_y', yy, episode)
            writer.add_scalar('te_z', zz, episode)
            if (xx+yy+zz) < best_loss:
                best_loss = xx+yy+zz 
                self.save("ckpt/obs1.pt")

            '''
            if (episode % 5 == 0):
                tqdm.write("[INFO] epi %05d | loss: %10.4f, %10.4f, %10.4f |" % (episode, Lx, Ly, Lz))
            if (episode % 20 == 0):
                xx, yy, zz = self.test(o_te, d_te, s_te)
                tqdm.write("[INFO] val loss: %10.4f, %10.4f, %10.4f" % (xx, yy, zz))
                if (xx+yy+zz) < best_loss:
                    best_loss = xx+yy+zz 
                    self.save("ckpt/obs.pt")
            '''
        writer.close()

    def test(self, o_te, d_te, s_te):
        length = o_te.shape[0] 
        Lx = Ly = Lz = 0
        for _ in range(0, length-batch_size, batch_size):
            o     = o_te[_:_+batch_size].to(device)
            d     = d_te[_:_+batch_size].to(device)
            s     = s_te[_:_+batch_size].to(device)
            s_, e = self.forward(o, d)
            loss  = self.loss(s_, s, e)
            Lx += loss[0].item(); Ly += loss[1].item(); Lz += loss[2].item()
        d = (length//batch_size)
        Lx /= d; Ly /= d; Lz /= d
        return Lx, Ly, Lz

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

def generate_data(total=10000):
    env = gym.make('ActivePerception-v0')
    data = []
    for _ in tqdm(range(total)):
        scene_data, obs = env.reset(False)
        state = get_state(scene_data)
        data.append((trans_rgb(obs['o']), trans_d(obs['d']), state))
        for j in range(1, 8):
            th  = np.pi/4*j 
            obs = env.step(th)
            s   = rotate_state(state, -th)
            data.append((trans_rgb(obs['o']), trans_d(obs['d']), s))
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
    #generate_data()
    dr = Derenderer() 
    dr.pretrain()
    #dr.visualize()
