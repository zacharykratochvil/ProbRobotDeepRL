import gym
import numpy as np
import math
import pybullet as p
from .ball import Ball
from .plane import Plane
from . import turtlebot
from .walls import Walls
import matplotlib.pyplot as plt
from typing import Generic
import cv2

class TurtleRLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super(TurtleRLEnv,self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.box.Box(low=0,
                                high=255, shape=(1,15000), dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random()

        if kwargs["env_type"] == "gui":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.bot = None
        self.goal = None
        self.walls = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()

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

        if dist_to_goal < self.goal.diameter/2:
            self.done = True
            reward = 50
        elif not is_valid:
            reward = -5
        else:
            reward = 0

        self.render(False)
        img_array = self.rendered_img.make_image(None,magnification=1/3.69)
        img_array = img_array[0][0:50,0:100,0:3] # crop out self and alpha
        observation = np.reshape(img_array,[1,-1])
        return observation, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

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
        
        self.render(False)
        img_array = self.rendered_img.make_image(None,magnification=1/3.69)
        img_array = img_array[0][0:50,0:100,0:3] # crop out self and alpha
        return np.reshape(img_array,[1,-1])

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

    def close(self):
        p.disconnect(self.client)