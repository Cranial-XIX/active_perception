import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn

from config import *
from operator import itemgetter

## general ####################################################################

def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)

def get_datetime():
    currentDT = datetime.datetime.now()
    return currentDT.strftime("%m%d%H%M")

## reward functions ###########################################################

def get_reward():
    pass

## module related #############################################################

class flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

## environment related ########################################################

def get_state(scene_data):
    state = np.zeros(dim_state)
    for obj in scene_data:
        idx, xyz = itemgetter('idx', '3d_coords')(obj)
        x, y, z  = xyz
        state[idx*dim_obj:(idx+1)*dim_obj] = [1,x,y,z]
    return torch.FloatTensor(state).unsqueeze(0)

def get_spherical_state(scene_data):
    state = np.zeros(dim_state)
    for obj in scene_data:
        idx, xyz = itemgetter('idx', '3d_coords')(obj)
        x, y, z  = xyz
        r   = np.sqrt(x**2+y**2+z**2)
        phi = np.arctan(y/x)
        th  = np.arccos(z/r)
        state[idx*dim_obj:(idx+1)*dim_obj] = [1, r, phi, th] 
    return torch.FloatTensor(state).unsqueeze(0)

def rotate_state(state, theta, device="cpu"):
    """
    rotate the cartesian coordinates counter-clockwise by theta (only x,y)
    state: [B, n_obj*dim_obj]
    """
    s  = state.clone().view(-1, n_obj, dim_obj)
    v  = s[:,:,0].unsqueeze(-1)
    xy = s[:,:,1:3]
    z  = s[:,:,3].unsqueeze(-1)
    R  = torch.FloatTensor([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]]).to(device)
    xy = xy.view(-1, 2).mm(R).view(-1, n_obj, 2)
    s  = torch.cat((v, xy, z), -1)
    return s

def trans_rgb(rgb):
    rgb = torch.FloatTensor(rgb.transpose(2,0,1)/255).unsqueeze(0)
    return rgb

def get_estimated_state(p, w):
    """
    @param  p: [B, K, dim_state]
    @param  w: [B, K]
    @return s: [B, dim_state]
    """
    ww = w.exp()
    ww = ww / (ww.sum(1, keepdim=True) + 1e-10)
    s = (p * ww.unsqueeze(2)).sum(1)
    return s
