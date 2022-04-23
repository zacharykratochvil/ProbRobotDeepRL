import pybullet as p
import os

class Ball:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), "structures", "large_ball.urdf")
        self.id = p.loadURDF(fileName=f_name,
                   basePosition=[pos[0], pos[1], 0],
                   physicsClientId=client)
        self.pos = pos
        
        bounding_box = p.getAABB(self.id, physicsClientId=client)
        self.diameter = bounding_box[0][1] - bounding_box[1][1] # difference in x coords

                