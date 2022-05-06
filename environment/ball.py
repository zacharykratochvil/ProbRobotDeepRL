import pybullet as p
import os

class Ball:
    def __init__(self, client, pos):
        f_name = os.path.join(os.path.dirname(__file__), "structures", "large_ball.urdf")
        self.pos = pos

        self.id = p.loadURDF(fileName=f_name,
                   basePosition=[pos[0], pos[1], 0],
                   physicsClientId=client)
        
        bounding_box = p.getAABB(self.id, physicsClientId=client)
        self.diameter = abs(bounding_box[0][1] - bounding_box[1][1]) # difference in x coords

        p.removeBody(self.id)
        self.id = p.loadURDF(fileName=f_name,
                   basePosition=[pos[0], pos[1], self.diameter/2],
                   physicsClientId=client)
                