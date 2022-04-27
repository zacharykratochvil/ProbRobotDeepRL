import gym
# from .turtle_rl_env import TurtleRLEnv
from .sham_test_env import ShamTestEnv

# gym.envs.register(
#      id='TurtleRLEnv-v0',
#      entry_point='environment:TurtleRLEnv',
# )
gym.envs.register(
     id='ShamTestEnv-v0',
     entry_point='environment:ShamTestEnv',
)