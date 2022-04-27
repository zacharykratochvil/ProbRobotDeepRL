import pybullet as p
import os
import math
import numpy as np

# define constants
FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3

def opposite_action(action):
    if action == FORWARD:
        return BACKWARD
    if action == BACKWARD:
        return FORWARD
    if action == LEFT:
        return RIGHT
    if action == RIGHT:
        return LEFT

# Turtlebot class handles structure and movement of turtlebot
class Turtlebot:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "structures", "turtlebot.urdf")
        self.id = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0],
                              physicsClientId=client)

    # actions are defined as constants in this module
    def apply_action(self, action):

        # get info about bot's whereabouts
        pos, quat = p.getBasePositionAndOrientation(self.id, self.client)
        pos = np.array(pos)
        _, unit_vec, vel = self.get_observation()
        unit_vec = np.array([unit_vec[0], unit_vec[1], 0])
        ang = np.array(p.getEulerFromQuaternion(quat))

        # teleport bot to given coords
        def teleport(new_pos, new_ang):
            p.resetBasePositionAndOrientation(self.id,
                        new_pos, new_ang, self.client)

        # define discrete actions
        drive_magnitude = .1 # meters
        turn_magnitude = 5*np.pi/180 # radians

        if action == FORWARD:
            new_pos = pos + drive_magnitude*unit_vec
            teleport(new_pos, quat)

        elif action == BACKWARD:
            new_pos = pos - drive_magnitude*unit_vec
            teleport(new_pos, quat)

        elif action == LEFT:
            ang[2] += turn_magnitude
            new_quat = p.getQuaternionFromEuler(ang)
            teleport(pos, new_quat)

        elif action == RIGHT:
            ang[2] -= turn_magnitude
            new_quat = p.getQuaternionFromEuler(ang)
            teleport(pos, new_quat)

        else:
            raise Exception(f"{action} is not a valid action, must be integer [0,3]")

    def get_observation(self):
        # Get the position and orientation of the bot in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.id, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        
        # Get the velocity of the bot
        vel = p.getBaseVelocity(self.id, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos, ori, vel)

        return observation