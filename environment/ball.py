import pybullet as p
import os

class Ball:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), "structures", "large_ball.urdf")
        p.loadURDF(fileName=f_name,
                   basePosition=[pos[0], pos[1], 0],
                   physicsClientId=client)

                