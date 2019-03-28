from clevr_envs.clevr import ActivePerceptionEnv
from clevr_envs.wrappers import Wrapper, DataCollector
from gym.envs.registration import register

register(
    id='ActivePerception-v0',
    entry_point='clevr_envs:ActivePerceptionEnv',
)
