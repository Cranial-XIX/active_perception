import copy
import json
import numpy as np
import numpy.random as rd
import os
import random
import sys

COLORS = ["gray", "red", "blue", "green"]
SHAPES = ["cube", "cylinder"]
SIZES  = ["large", "small"]
nC = len(COLORS); nSH = len(SHAPES); nCxSH=nC*nSH
sz2h = {"large": 1.0, "small": 0.5}

def get_all_objects():
    large_objects, small_objects = {}, {}
    cnt = 0
    for color in COLORS:
        for shape in SHAPES:
            large_obj = {
                "color": color,
                "size": "large",
                "rotation": 0,
                "shape": shape,
                "3d_coords": [
                    0, 0, 0
                ],
                "material": "metal",
                "occlude": 1,
                "area": 100,
            }

            small_obj = {
                "color": color,
                "size": "small",
                "rotation": 0,
                "shape": shape,
                "3d_coords": [
                    0, 0, 0
                ],
                "material": "metal",
                "occlude": 0,
                "area": 100,
            }
            large_objects[cnt] = large_obj
            small_objects[cnt] = small_obj
            cnt += 1
    return large_objects, small_objects

large_objects, small_objects = get_all_objects()

def sample_2d_coords(xmin, xmax, ymin, ymax, n, d, reject_band=None):
    """
    Sample n 2d coordinates uniformly in range [xmin, xmax] x [ymin, ymax],
    so that they keep a distance d away from each other
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param num:
    :param d:
    :param reject_band ([rej_xmin, rej_xmax, rej_ymin, rej_ymax])
    :return: xs and ys, two lists
    """
    require_resample = True
    [rej_xmin, rej_xmax, rej_ymin, rej_ymax] = reject_band
    while require_resample:
        xs, ys = [], []
        for i in range(n):
            sample_cnt = 0
            # require_resample = False
            # sample_cnt = 0
            def safe(x, y):
                position_valid = reject_band is None or reject_band is not None and not (x>rej_xmin and x<rej_xmax and y>rej_ymin and y<rej_ymax)
                if not position_valid:
                    return False
                for _x, _y in zip(xs, ys):
                    # if reject_band is None: # no constraints on the position
                    if np.sqrt((x-_x)**2+(y-_y)**2) < d:
                        return False
                return True
            x, y = rd.uniform(xmin, xmax), rd.uniform(ymin, ymax)

            while not safe(x, y) and sample_cnt < 100:
                x, y = rd.uniform(xmin, xmax), rd.uniform(ymin, ymax)
                sample_cnt += 1

            if sample_cnt == 100:
                break

            xs, ys = xs + [x], ys + [y]

        require_resample = False if sample_cnt < 100 else True
    # print('sample {} stacks, get len {}'.format(n, len(xs)))
    return xs, ys

def gen_stack(rest_pool, x, y, num, size='large'):
    """
    generate a stack of *num* objects at position x, y from rest_pool
    rest_pool: list of rest object indices
    """
    stack = []
    single_height = sz2h[size]
    heights = [single_height, single_height*3, single_height*5, single_height*7, single_height*9]

    for _ in range(num):
        obj_base_ind = random.choice(rest_pool)
        obj_base = large_objects[obj_base_ind] if size == 'large' else small_objects[obj_base_ind]
        while obj_base['shape'] == 'sphere':
            # do not have sphere
            obj_base_ind = random.choice(rest_pool)
            obj_base = large_objects[obj_base_ind] if size == 'large' else small_objects[obj_base_ind]
        obj = copy.copy(obj_base)
        obj["3d_coords"] = [x, y, heights[_]]
        stack.append(obj)
        rest_pool.remove(obj_base_ind)
    return rest_pool, stack

def sample_front_stacks():
    num_stacks = rd.randint(1, 3) # scene has 1 to 2 stacks
    x_stacks, y_stacks = sample_2d_coords(0+0.4, 12-0.4, -0.6, 2, num_stacks, 1, reject_band=[5, 7, -1.5, 1.5])
    total_obj_num = 10000
    rest_pool = list(large_objects.keys())
    objects = []
    total_obj_num = 0
    num_per_stack = {}
    for i in range(num_stacks):
        num_obj_this_stack = rd.randint(1, 3)
        rest_pool, stack = gen_stack(rest_pool, x_stacks[i], y_stacks[i], num_obj_this_stack, size='large')
        objects.extend(stack)
        total_obj_num += num_obj_this_stack
        num_per_stack[(x_stacks[i], y_stacks[i])] = num_obj_this_stack

    return objects, num_per_stack

def sample_back_stacks(stacks, num_per_stack):
    objects = []

    rest_pool = list(small_objects.keys())
    num_front = num_back = 0
    for location in num_per_stack.keys():
        num_front_this_stack = num_per_stack[location]
        num_front += num_front_this_stack
        # exist_occ = random.random() > 0.5 and num_front_this_stack > 1 # flag to tell if the stack has occluded objects
        exist_occ = True
        if exist_occ:
            num_occ_this_stack = rd.randint(1, num_front_this_stack+1)
            num_back += num_occ_this_stack
            x, y = sample_2d_coords(location[0]-.4, location[0]+.4, location[1]-1.6, location[1]-1.5, 1, 1, reject_band=[5, 7, -1.5, 1.5])
            # x, y = sample_2d_coords(location[0]-.4, location[0]+.4, location[1]-1.6, location[1]-1.5, 1, 1.5)
            rest_pool, stack = gen_stack(rest_pool, x[0], y[0], num_occ_this_stack, size='small')
            objects.extend(stack)
    print("[INFO] sampled %d front %d back" % (num_front, num_back))

    return objects

def gen_scene(idx, stacks, back):
    scene = {
        "image_index": idx,
        "image_filename": "%05d_image.png" % idx,
        "objects": back + stacks
    }
    return scene

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("[INFO] usage: [NUM_SCENE]")
        sys.exit(0)

    N = int(sys.argv[1])

    scenes = []
    stacks = []
    for _ in range(N):
        stacks, num_per_stack = sample_front_stacks()
        backs = sample_back_stacks(stacks, num_per_stack)
        scenes.append(gen_scene(_, stacks, backs))

    json_dict = {
        "info": "scene consists of front/back objects",
        "scenes": scenes
    }
    json.dump(json_dict, open(os.path.join("clevr_envs", "scene.json"), 'w'), indent=4)
