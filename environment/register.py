import gym
import environment

#######
# register
#   -tells gym where to look for environments
#   -adds only the relevant environment to the module's namespace
#######
def register(id):

    if id == 'TurtleRLEnv-v0':
        from .turtle_rl_env import TurtleRLEnv
        setattr(environment, "TurtleRLEnv", TurtleRLEnv)
        gym.envs.register(
            id=id,
            entry_point='environment:TurtleRLEnv',
        )

    elif id == 'ShamTestEnv-v0':
        from .sham_test_env import ShamTestEnv
        setattr(environment, "ShamTestEnv", ShamTestEnv)
        gym.envs.register(
            id=id,
            entry_point='environment:ShamTestEnv',
        )

    elif id == 'SimpleTurtleRLEnv-v0':
        from .simple_turtle_rl_env import SimpleTurtleRLEnv
        setattr(environment, "SimpleTurtleRLEnv", SimpleTurtleRLEnv)
        gym.envs.register(
            id=id,
            entry_point='environment:SimpleTurtleRLEnv',
        )