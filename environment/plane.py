import pybullet as p
import os

class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), "structures", "plane.urdf")
        p.loadURDF(fileName=f_name,
                   basePosition=[0, 0, -.125/2],
                   physicsClientId=client)