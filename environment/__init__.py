import gym
from .turtle_rl_env import TurtleRLENV

gym.envs.register(
     id='TurtleRLENV-v0',
     entry_point='environment:TurtleRLENV',
)