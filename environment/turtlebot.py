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
        # Expects action to be two dimensional
        throttle1, throttle2 = action

        # Clip throttle and steering angle to reasonable values
        throttle1 = min(max(throttle1, 0), 1)
        throttle2 = min(max(throttle2, 0), 1)
        throttle = np.array([throttle1, throttle2])

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        self.joint_speed = np.max(np.vstack([self.joint_speed, np.zeros(2)]),0)

        # Set the velocity of the wheel joints directly
        for i in range(len(self.drive_joints)):
            p.setJointMotorControlArray(
                bodyUniqueId=self.car,
                jointIndices=[self.drive_joints[i]],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[self.joint_speed[i]],
                forces=[1.2],
                physicsClientId=self.client)

    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.car, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.car, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        return observation