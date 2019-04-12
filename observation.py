import clevr_envs
import gym
import torch
import torch.nn
import torch.nn.functional as F

from utils import *

class Observation(nn.Module):
    def __init__(self):
        super(Observation, self).__init__()
        self.derenderer = nn.Sequential(
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
            nn.Linear(dim_hidden, dim_obs)
        )

    def forward(self, x):
        pass

def generate_data(total=100):
    env = gym.make('ActivePerception-v0')
    data = []
    for _ in range(total):
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

if __name__ == "__main__":
    generate_data(1000)
