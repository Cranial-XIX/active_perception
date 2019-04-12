import clevr_envs
import copy
import json
import gym
import numpy as np
import os
import sys

from PIL import Image
from config import *
from generate_initial_scene import gen_obj 
from utils import *

def get_scene(data):
    data = data.astype('float')
    objects = []
    for idx in range(n_obj):
        if data[4*idx] < 0.5:
            continue
        else:
            i = 4*idx
            objects.append(gen_obj(idx, data[i], data[i+1], data[i+2]))

    return {
        "objects": objects
    }

def visualize_b(target, estimates, test_name="tmp", is_rnn=True):
    suffix = "rnn" if is_rnn else "obs"
    test_scenes = {}
    test_scenes['info'] = test_name 
    scenes = []
    names  = []
    scenes.append(get_scene(target))
    names.append("target")
    for i, e in enumerate(estimates):
        scenes.append(get_scene(e))
        names.append("est-%d" % (i+1))
    total = len(scenes)
    test_scenes['scenes'] = scenes 
    json.dump(test_scenes, open(os.path.join("clevr_envs", "scene_test.json"), 'w'), indent=4)
    clevr_envs.clevr.TEST = True
    env = gym.make('ActivePerception-v0')
    check_path("img/"+test_name)
    for _, name in zip(range(total), names):
        state, obs = env.reset(_)
        Image.fromarray(obs['a'][::-1]).save("img/%s/%s-%s.png" % (test_name, suffix, name))

def visualize_o(objects):
    scenes = [{'objects': objects}]
    test_scenes = {'scenes': scenes}
    json.dump(test_scenes, open(os.path.join("clevr_envs", "scene_test.json"), 'w'), indent=4)
    clevr_envs.clevr.TEST = True
    clevr_envs.clevr.TRANSPARENT = True
    env = gym.make('ActivePerception-v0')
    env.test = True
    check_path("img/obs")
    _, obs = env.reset(0)
    Image.fromarray(obs['a'][::-1]).save("img/obs/predicted.png")
