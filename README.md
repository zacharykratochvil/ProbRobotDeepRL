# ProbRobotDeepRL
*Deep reinforcement learning project for our Probabilistic Robotics class at Tufts.*  
Authors: Niko Ciminelli, Bharat Kesari, Zachary Kratochvil

## About  
This is code for our simulation of a ball-finding task with a Turtlebot.

## Configuring Environment
` pip install poetry `  
` poetry install `

## Running Simulation from Pre-trained Model
` poetry run python main.py --gui --num-envs 1 --train `

## Training
` poetry run python main.py --num-envs 4 --train `

## Additional Information
Notes: We only use the simulator to
simulate optics. Our observation space is images and PyBullet
renders them nicely. We're using discrete position-control
actions in our robot so we best simulate this by teleporting.

## Built using:
* https://github.com/GerardMaggiolino/TRPO-Implementation
* https://github.com/erwincoumans/pybullet_robots
* https://github.com/ericyangyu/PPO-for-Beginners
* https://arxiv.org/abs/1707.06347
