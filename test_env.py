import clevr_envs
import copy
import json
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from PIL import Image

def test():
    env = gym.make('ActivePerception-v0')
    state, obs = env.reset()
    print(state)
    #camera = obs['camera'][0][::-1]
    #birdview = obs['birdview'][0][::-1]
    #Image.fromarray(camera).save("camera.png")
    #Image.fromarray(birdview).save("birdview.png")
    #Image.fromarray(agentview).save("view.png")
    n = 8
    for i in range(n):
        theta = 2*np.pi / 8 * i
        obs = env.step(theta)
        Image.fromarray(obs['a'][::-1]).save("img/view"+str(i)+"pi_over_4.png")

    '''
    while True:
        env.render()
        env.step([0.0,0.1,0,-10])
        #env.step(env.action_space.sample())
    '''

def test2():
    env = gym.make('ActivePerception-v0')
    _, o = env.reset()
    rgb, d = o['o'], o['d']
    Image.fromarray(rgb).save("rgb.png")
    plt.imshow(d)
    plt.savefig("depth.png", cmap='gray')
    plt.close()
    print(d.shape)

def test_rotation():
    objects = []
    xy = np.array([1.5, 0]).reshape(1,2)
    for _ in range(8):
        theta = np.pi/4*_
        R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])
        s = np.matmul(xy, R) 
        objects.append({
            "idx": 0,
            "3d_coords": [s[0,0],s[0,1], 0.25],
            "material" : "metal",
        })

    scenes = [{'objects': objects}]
    test_scenes = {'scenes': scenes}
    json.dump(test_scenes, open(os.path.join("clevr_envs", "scene.json"), 'w'), indent=4)
    env = gym.make('ActivePerception-v0')
    _, obs = env.reset()
    Image.fromarray(obs['o']).save("0.png")
    obs    = env.step(np.pi/2)
    Image.fromarray(obs['o']).save("1.png")
    env.close()

if __name__ == '__main__':
    #test()
    test_rotation()
