import pybullet as p
import os
import numpy as np


class Walls:
    def __init__(self, client):
        short_f_name = os.path.join(os.path.dirname(__file__), "structures", 'wall-short.urdf')
        long_f_name = os.path.join(os.path.dirname(__file__), "structures", 'wall-long.urdf')

        # load and remove urdfs to get their length
        test_short_wall = p.loadURDF(fileName=short_f_name,
                   basePosition=[0, 0, 0],
                   baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=client)
        
        self.short_wall_half_length = np.abs(p.getAABB(test_short_wall, physicsClientId=client)[0][0])
        self.short_wall_half_thickness = np.abs(p.getAABB(test_short_wall, physicsClientId=client)[0][2])
        self.short_wall_half_height = np.abs(p.getAABB(test_short_wall, physicsClientId=client)[0][1])
        
        p.removeBody(test_short_wall)
        
        translation_dist = self.short_wall_half_length + self.short_wall_half_thickness

        # add the four walls
        self.wall = np.zeros(4, int)
        self.wall[0] = p.loadURDF(fileName=short_f_name,
                   basePosition=[0, translation_dist, self.short_wall_half_height],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,0]),
                   physicsClientId=client)
        self.wall[1] = p.loadURDF(fileName=short_f_name,
                   basePosition=[0, -translation_dist, self.short_wall_half_height],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,0]),
                   physicsClientId=client)
        self.wall[2] = p.loadURDF(fileName=long_f_name,
                   basePosition=[translation_dist, 0, self.short_wall_half_height],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]),
                   physicsClientId=client)
        self.wall[3] = p.loadURDF(fileName=long_f_name,
                   basePosition=[-translation_dist, 0, self.short_wall_half_height],
                   baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,np.pi/2]),
                   physicsClientId=client)