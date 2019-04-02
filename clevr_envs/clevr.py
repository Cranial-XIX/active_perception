import copy
import cv2
import gym
import json
import numpy as np
import os
import xml
import xml.etree.ElementTree as ET

from gym import utils
import sys
sys.path.append("..")
from config import *

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, " \
        "and also perform the setup instructions here:" \
        "https://github.com/openai/mujoco-py/.)".format(e))

CAMERA_RADIUS = 9.5
CAMERA_HEIGHT = 3
SCENE_ID      = 0
SEGMENTATION  = 0     # 0 default condition, 1 segmentation condition
TRANSPARENT   = False

BASE_XML_FILE   = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'base.xml'))
SCENE_JSON_FILE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'scene.json'))
FINAL_XML_FILE  = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'final.xml'))

properties_data = {
  "shapes": {
    "cube"    : "SmoothCube_v2",
    "sphere"  : "Sphere",
    "cylinder": "SmoothCylinder"
  },
  "colors": {
    "gray"  : [87, 87, 87],
    "red"   : [173, 35, 35],
    "blue"  : [42, 75, 215],
    "green" : [29, 105, 20],
    "brown" : [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan"  : [41, 208, 208],
    "yellow": [255, 238, 51]
  },
  "materials": {
    "rubber": "Rubber",
    "metal" : "MyMetal"
  },
  "sizes": {
    "large": 0.25,
    "small": 0.125
  }
}

## Utils #######################################################################
def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7-j))*((i & (1 << (3*j))) >> (3*j))
            g = g + (1 << (7-j))*((i & (1 << (3*j+1))) >> (3*j+1))
            b = b + (1 << (7-j))*((i & (1 << (3*j+2))) >> (3*j+2))

        cmap[i, :] = np.array([r, g, b])

    return cmap

def array_to_string(array):
    """
        Converts a numeric array into the string format in mujoco
        [0, 1, 2] => "0 1 2"
    """
    return ' '.join(['{}'.format(x) for x in array])

def string_to_array(string):
    """
        Converts a array string in mujoco xml to np.array
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(' ')])

def set_alpha(node, alpha = 0.1):
    """
        Sets all a(lpha) field of the rgba attribute to be @alpha
        for @node and all subnodes
        used for managing display
    """
    for child_node in node.findall('.//*[@rgba]'):
        rgba_orig = string_to_array(child_node.get('rgba'))
        child_node.set('rgba', array_to_string(list(rgba_orig[0:3]) + [alpha]))

def joint(**kwargs):
    """
        Create a joint tag with attributes specified by @**kwargs
    """
    element = ET.Element('joint', attrib=kwargs)
    return element

################################################################################
#                                                                              #
# Env for active perception                                                    #
#                                                                              #
################################################################################
class ActivePerceptionEnv(gym.Env, utils.EzPickle):

    def __init__(self, control_freq=50):
        # load scene data
        clevr_data = json.load(open(SCENE_JSON_FILE, 'r'))
        self.scenes = clevr_data['scenes']
        self.total_scenes = len(self.scenes)
        self.control_freq = control_freq

    def start(self, scene_id):
        self.scene_data = self.scenes[scene_id]

        # load the mesh and add the objects
        self.tree = ET.parse(BASE_XML_FILE)
        self.root = self.tree.getroot()
        self.worldbody = self.root.find('worldbody')

        self.obj_xquat = dict()

        # store timestep
        self.mj_timestep = float(self.root.find('option').get('timestep'))

        clevr_to_mujoco_types = {
            'cube': 'box',
            'cylinder': 'cylinder',
            'sphere': 'sphere'
        }

        list_objects = self.scene_data['objects']
        n_objects = len(list_objects)
        cmap = colormap(n_objects + 1)

        for obj_id, obj in enumerate(list_objects):
            pos = np.array(obj['3d_coords'])
            idx = obj['idx']

            body = ET.Element('body', attrib={
                'pos': array_to_string(pos),
                'name': str(idx)
            })

            # object size
            size = [sz2h[CLEVR_OBJECTS[idx][2]]] * 3

            # object color
            color = np.array(properties_data['colors'][CLEVR_OBJECTS[idx][0]])/255.0

            alpha = 0.2 if TRANSPARENT else 1.0
            color = np.append(color, [alpha])

            geom = {
                'pos': '0 0 0',
                'type': clevr_to_mujoco_types[CLEVR_OBJECTS[idx][1]],
                'rgba': array_to_string(color),
                'size': array_to_string(size),
                'condim': '4',
                'name': str(idx),
                'solimp': '0.99 0.99 0.001',
                'solref': '0.001 1',
                'friction': '1.0 0.5 0.5',
                'mass': '.1',
            }

            body.append(ET.Element('geom', attrib=geom))

            joint = {
                'name': str(idx),
                'type': 'free',
                'damping': '.01'
            }
            body.append(ET.Element('joint', attrib=joint))

            site = {
                'pos': '0 0 0',
                'size': '0.002 0.002 0.002',
                'rgba': '1 0 0 1',
                'type': 'sphere',
                'name': str(idx),
            }
            body.append(ET.Element('site', attrib=site))

            self.worldbody.append(body)

        with open(FINAL_XML_FILE, 'w') as f:
            xml_str = ET.tostring(self.root, encoding='unicode')
            parsed_xml = xml.dom.minidom.parseString(xml_str)
            xml_str = parsed_xml.toprettyxml(newl='')
            f.write(xml_str)

        # set n_substeps automatically depending on control frequency
        n_substeps = int(1.0 / (self.control_freq * self.mj_timestep))
        model = mujoco_py.load_model_from_path(FINAL_XML_FILE)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None

    def _check_contact(self):
        return False

    def _get_obs(self):
        camera_obs = self.sim.render(camera_name="external_camera_0",
                                     width=512,
                                     height=512,
                                     depth=True)

        birdview_obs = self.sim.render(camera_name="external_camera_1",
                                     width=512,
                                     height=512,
                                     depth=True)

        agent_obs = self.sim.render(camera_name="agent_camera",
                                     width=64,
                                     height=64,
                                     depth=True)
        o = agent_obs[0].copy()
        #o = cv2.GaussianBlur(o, (3, 3), 3)
        return {
            'camera': (camera_obs[0].copy(), camera_obs[1].copy()),
            'birdview' : (birdview_obs[0].copy(), birdview_obs[1].copy()),
            'agentview' : (agent_obs[0].copy(), agent_obs[1].copy()),
            'o': o,
        }

    def reset(self):
        self.start(scene_id=np.random.randint(self.total_scenes))
        return (copy.deepcopy(self.scene_data['objects']), self.step(0))

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        return self.viewer

    def step(self, action):
        theta = action % (2*np.pi)
        x     = np.cos(theta)*CAMERA_RADIUS
        y     = np.sin(theta)*CAMERA_RADIUS
        z     = CAMERA_HEIGHT
        self.sim.data.set_mocap_pos('camera_mover', np.array([x,y,z]))

        qx, qy, qz = 0, 0, 1
        alpha = theta
        cos_a, sin_a = np.cos(alpha/2), np.sin(alpha/2)
        self.sim.data.set_mocap_quat('camera_mover', np.array([
                cos_a,
                sin_a*qx,
                sin_a*qy,
                sin_a*qz]))
        self.sim.forward()
        return self._get_obs()
