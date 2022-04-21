import pybullet as p
import os
import math
import numpy as np


class Car:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "structures", "turtlebot.urdf")
        print(f_name)
        self.car = p.loadURDF(fileName=f_name,
                              basePosition=[0, 0, 0],
                              physicsClientId=client)

        # Joint indices as found by p.getJointInfo()
        self.drive_joints = [0, 1]
        # Joint speed
        self.joint_speed = np.array([0, 0])
        # Drag constants
        self.c_rolling = 0.2
        self.c_drag = 0.01
        # Throttle constant increases "speed" of the car
        self.c_throttle = 20

    def get_ids(self):
        return self.car, self.client

    def apply_action(self, action):
        # Expects action to be an integer [0, 3]

        # get info about car's whereabouts
        pos, qat = p.getBasePositionAndOrientation(self.car, self.client)
        pos = np.array(pos)
        _, unit, vel = self.get_observation()
        unit = np.array([unit[0], unit[1], 0])
        ang = p.getEulerFromQuaternion(qat)

        # teleport car to given coords
        def teleport(new_pos, new_ang):
            p.resetBasePositionAndOrientation(self.car,
                        new_pos, new_ang, self.client)

        # define discrete actions
        if action == 0: # forward
            teleport(pos + unit, qat)
        elif action == 1: # backward
            teleport(pos - unit, qat)
        elif action == 2: # turn left
            pass
        elif action == 3: # turn right
            pass
        else:
            raise Exception(f"{action} is not a valid action, must be integer [0,3]")

    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos, ori, vel)

        return observation