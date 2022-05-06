import gym
import numpy as np
import scipy.ndimage as sp_img

class ShamTestEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        super(ShamTestEnv,self).__init__()
        # defines the expected input and output of the environment
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.box.Box(low=0,
                                high=1, shape=(3,100,100), dtype=np.float32)
        
        # global state of environment
        self.observation = None
        self.done = False

    ######
    # advances the simulation one step 
    #   inputs:
    #       action - the action to take for this step
    #   
    #   returns:
    #       observation - the input to the deep RL network
    #       reward - the reward accumulated at this step
    #       done - whether or not the simulation has ended
    #       info - dictionary of debugging info
    #######
    def step(self, action):
        
        self.make_observation()
        reward = 50
        # randomly become done 10% of the time, otherwise keep state
        self.done = self.done or (np.random.randint(0,10) == 0)
        
        return self.observation, reward, self.done, dict()

    #######
    # sets the random seed of the environment to the specified number
    #######
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    ######
    # resets the environment to its original state
    # returns the current observation, just as in the function "step" above
    ######
    def reset(self):
        self.done = False
        self.make_observation()
        return self.observation

    ######
    # test environment observations are generated randomly
    ######
    def make_observation(self):
        # randomly generate input the same size as we get from our actual camera
        self.observation = np.random.rand(3,480,640)
        #change size obs = transoformation... to 50,150
        # (1, 0.2083333333, 0.234375) 240x640 to 50x150
        # (1, 0.41666667, 0.15625) 480x640 to 100x100!!!!!!!!Yash version
        self.observation = sp_img.zoom(self.observation, zoom = (1, 0.2083333333, 0.15625), order=1)

    ######
    # would output an image of the environment, but this is a test so it is not used
    ######
    def render(self, plot=True):
        pass