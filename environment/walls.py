import pybullet as p
import os
import numpy as np


class Walls:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), "structures", 'wall.urdf')

        # load and remove urdf to get its length
        test_wall = p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, 0],
                   baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=client)
        wall_half_length = p.getAABB(test_wall)[0][0]
        p.removeBody(test_wall)

        # add the four walls
        self.wall = np.zeros(4)
        self.wall[0] = p.loadURDF(fileName=f_name,
                   basePosition=[0, wall_half_length, 0],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,0]),
                   physicsClientId=client)
        self.wall[1] = p.loadURDF(fileName=f_name,
                   basePosition=[0, -wall_half_length, 0],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,0]),
                   physicsClientId=client)
        self.wall[2] = p.loadURDF(fileName=f_name,
                   basePosition=[wall_half_length, 0, 0],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]),
                   physicsClientId=client)
        self.wall[3] = p.loadURDF(fileName=f_name,
                   basePosition=[-wall_half_length, 0, 0],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]),
                   physicsClientId=client)