"""
This file contains a wrapper for saving simulation states.
This is useful for collecting demonstrations.
"""

from clevr_envs.wrappers.wrapper import Wrapper
import os
import time
import numpy as np

class DataCollector(Wrapper):

    def __init__(self, env, directory, collect_freq=1, flush_freq=1000):
        """
        :param env: The environment to monitor.
        :param directory: Where to store collected data.
        :param collect_freq: How often to save simulation state, in terms of environment steps.
        :param flush_freq: How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states
        self.states = [] 

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            print("DataCollector: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred 
        self.has_interaction = False

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode. 
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split('.')
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert(not os.path.exists(self.ep_directory))
        print("DataCollector: making new directory at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, 'model.xml')
        with open(xml_path, 'w') as f:
            xml_str = self.env.sim.model.get_xml()
            f.write(xml_str)
        # self.env.task.save_model(xml_path)

    def _flush(self):
        """
        Method to flush internal state to disk. 
        """
        t1, t2 = str(time.time()).split('.')
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        np.savez(state_path, states=np.array(self.states))
        self.states = []

    def reset(self):
        ret = super().reset()
        self._start_new_episode()
        return ret

    def step(self, action):

        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            # state = self.env.physics.state()
            state = self.env.sim.get_state().flatten()
            self.states.append(state)

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret





