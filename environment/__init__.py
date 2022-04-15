import gym
from .turtle_rl_env import TurtleRLEnv

gym.envs.register(
     id='TurtleRLEnv-v0',
     entry_point='environment:TurtleRLEnv',
)