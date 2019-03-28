"""
This file contains the base wrapper class for Mujoco environments.
Wrappers are useful for data collection and logging. Highly recommended.
"""

from gym import Env

class Wrapper(Env):
    env = None

    def __init__(self, env):
        self.env = env
        self.sim = self.env.sim
        self.done = False

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        env = self.env
        while True:
            if isinstance(env, Wrapper):
                if env.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                env = env.env
            else:
                break

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        orig_attr = self.env.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.env:
                    return self
                return result
            return hooked
        else:
            return orig_attr
        




