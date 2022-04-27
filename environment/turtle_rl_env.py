import gym
import numpy as np
import math
import pybullet as p
from .ball import Ball
from .plane import Plane
from . import turtlebot
from .walls import Walls
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

class TurtleRLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super(TurtleRLEnv,self).__init__()
        # defines the expected input and output of the environment
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.box.Box(low=0,
                                high=1, shape=(3,50,150), dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random()

        # select whether to show pybullet's inbuilt gui
        if kwargs["gui"] == True:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        # initialize simulation state
        self.bot = None
        self.goal = None
        self.walls = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None

    ######
    # advances the simulation one step, calculating the reward
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
        # Feed action to the bot and get observation of bot's state
        self.bot.apply_action(action)
        is_valid = self.validate_position()
        if not is_valid: # undo invalid actions
            self.bot.apply_action(turtlebot.opposite_action(action))

        p.stepSimulation()
        bot_ob, _, _ = self.bot.get_observation()

        # Compute reward for reaching goal, punishment for hitting wall
        dist_to_goal = math.sqrt(((bot_ob[0] - self.goal.pos[0]) ** 2 +
                                  (bot_ob[1] - self.goal.pos[1]) ** 2))

        reward = -1
        if dist_to_goal < self.goal.diameter:
            self.done = True
            reward += 1000
        elif not is_valid:
            reward += -10

        observation = self.make_observation()
                
        return observation, reward, self.done, dict()

    #######
    # sets the random seed of the environment to the specified number
    #######
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    ######
    # resets the environment to its original state
    # randomizes position of goal
    # returns the current observation, just as in the function "step" above
    ######
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and bot
        Plane(self.client)
        self.walls = Walls(self.client)
        self.bot = turtlebot.Turtlebot(self.client)

        # Set the goal to a random target
        x = (self.np_random.uniform(.5, 1) if self.np_random.randint(2) else
             self.np_random.uniform(-1, -.5))
        y = (self.np_random.uniform(.5, 1) if self.np_random.randint(2) else
             self.np_random.uniform(-1, -.5))
        goal_pos = (x, y)
        self.done = False

        # Visual element of the goal
        self.goal = Ball(self.client, goal_pos)

        # Get observation to return
        pos, ori, vel = self.bot.get_observation()

        self.prev_dist_to_goal = math.sqrt(((pos[0] - goal_pos[0]) ** 2 +
                                           (pos[1] - goal_pos[1]) ** 2))
        
        observation = self.make_observation()
        return observation

    ######
    # constructs an image of the environment using pybullet
    # returns the image as well as storing it in self.rendered_img
    ######
    def render(self, plot=True):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(self.bot.id, self.client)]
        pos[2] = 0.3148

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Add noise and display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))

        self.rendered_img.set_data(frame)
        
        if plot == True:
            plt.draw()
            plt.pause(.00001)
        
        return frame


    ##########
    # For discrete environment we override much of the physics engine,
    # so this function is necessary to check for collisions
    #
    # returns: True if and only if bot's position is valid (no collisons)
    #          False otherwise 
    ##########
    def validate_position(self):

        #check for collison with walls
        pts = []
        for i in range(len(self.walls.wall)):
            pts.extend(p.getContactPoints(bodyA=self.bot.id, bodyB=self.walls.wall[i], physicsClientId=self.client))
        if len(pts) > 0:
            return False

        #check for being outside walls
        pos, quat = p.getBasePositionAndOrientation(self.bot.id, physicsClientId=self.client)
        pos = np.asarray(pos)
        env_radius = self.walls.wall_half_length

        if np.any(pos[0:2] > env_radius):
            return False
        elif np.any(pos[0:2] < -env_radius):
            return False
        else:
            return True

    ######
    # process rendered image into format of observation
    ######
    def make_observation(self):
        # update image
        self.render(False)
        # scale to ~480x640 the size of the robot's image
        img_array = self.rendered_img.make_image(None,magnification=1.3)
        #crop out robot and reorder axes for pytorch
        observation = np.transpose(img_array[0][0:240,0:640,0:3],[2,0,1])
        #change size obs = transoformation... to 50,150
        # (1, 0.2083333333, 0.234375) 240x640 to 50x150
        observation = zoom(observation, zoom = (1, 0.2083333333, 0.234375), order=1)
        # normalize for 256-bit color
        observation = observation/255
        return observation

    ######
    # let go of pybullet's resources
    ######
    def close(self):
        p.disconnect(self.client)