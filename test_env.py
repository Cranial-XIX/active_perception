import clevr_envs
import copy
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

if __name__ == '__main__':
    #test()
    test2()
