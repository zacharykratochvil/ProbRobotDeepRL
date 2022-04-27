import gym
import numpy as np
from scipy.ndimage import zoom

class ShamTestEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        super(ShamTestEnv,self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.box.Box(low=0,
                                high=1, shape=(3,50,150), dtype=np.float32)
        
        self.observation = None
        self.done = False

    def step(self, action):
        self.observation = np.random.rand(3,240,640)
        #zoom
        self.observation = zoom(self.observation,zoom = (1, 0.2083333333, 0.234375), order=1)
        print('*'*30)
        print('step', self.observation.shape)
        print('*'*30)
        reward = 50
        self.done = self.done or (np.random.randint(0,10) == 0)
        return self.observation, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.observation = np.random.rand(3,240,640)
        #change size obs = transoformation... to 50,150
        self.observation = zoom(self.observation,zoom = (1, 0.2083333333, 0.234375), order=1)# (1, 0.2083333333, 0.234375) 240x640 to 50x150
        print('*'*30)
        print('reset', self.observation.shape)
        print('*'*30)
        return self.observation

    def render(self, plot=True):
        pass