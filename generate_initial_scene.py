import copy
import json
import numpy as np
import numpy.random as rd
import os
import random
import sys

from config import *

def gen_obj(idx, x, y, z):
    return {
        "idx"      : int(idx),
        "3d_coords": [x, y, z],
        "material" : "metal",
    }

def random_xy():
    x = np.random.rand()*(XMAX-XMIN)+XMIN
    y = np.random.rand()*(YMAX-YMIN)+YMIN
    return x, y

def safe(x, y, xs, ys):
    if len(xs) == 0:
        return 1
    xx = np.array(xs) - x
    yy = np.array(ys) - y
    is_safe = np.sum(np.sqrt(np.square(xx) + np.square(yy)) > D_AWAY) >= len(xs)
    return is_safe
 
def sample_2d_coords(n):
    xs, ys = [], []
    i = 0
    while i < n:
        x, y = random_xy()
        if safe(x, y, xs, ys):
            i += 1
            xs.append(x)
            ys.append(y)
    return xs, ys

def gen_scene(idx):
    xs, ys = sample_2d_coords(n_obj)
    #print("[INFO] sampled coordinates %s" % str(xys))
    objects = []
    indices = np.random.permutation(n_obj)
    for i, x, y in zip(indices, xs, ys):
        obj = gen_obj(
                i,
                x,
                y,
                sz2h[CLEVR_OBJECTS[i][2]]
                )
        objects.append(obj)

    """
    for x, y in xys:
        # generate a stack
        n_obj = np.random.randint(1, TALLEST+1)
        curr_z = 0
        for _ in range(n_obj):
            sz = random.choice(SIZES)
            curr_z += sz2h[sz]
            obj = gen_obj(
                    random.choice(COLORS),
                    sz,
                    random.choice(SHAPES),
                    x,
                    y,
                    curr_z
                    )
            curr_z += sz2h[sz]
            objects.append(obj)
    """

    scene = {
        "image_index": idx,
        "image_filename": "%05d_image.png" % idx,
        "objects": objects
    }
    return scene

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[INFO] usage: [NUM_SCENE]")
        sys.exit(0)

    N = int(sys.argv[1])

    scenes = []
    for _ in range(N):
        scenes.append(gen_scene(_))

    json_dict = {
        "info": "scene consists of front/back objects",
        "scenes": scenes
    }
    json.dump(json_dict, open(os.path.join("clevr_envs", "scene.json"), 'w'), indent=4)
