import clevr_envs
import copy
import gym
import numpy as np
import os
import sys

from PIL import Image

def test():
    env = gym.make('ActivePerception-v0')
    obs = env.reset()
    #camera = obs['camera'][0][::-1]
    #birdview = obs['birdview'][0][::-1]
    #Image.fromarray(camera).save("camera.png")
    #Image.fromarray(birdview).save("birdview.png")
    #Image.fromarray(agentview).save("view.png")
    n = 8
    for i in range(n):
        theta = 2*np.pi / 8 * i
        obs = env.step(theta)
        Image.fromarray(obs['o'][::-1]).save("img/view"+str(i)+"pi_over_4.png")

    '''
    while True:
        env.render()
        env.step([0.0,0.1,0,-10])
        #env.step(env.action_space.sample())
    '''

if __name__ == '__main__':
    test()
