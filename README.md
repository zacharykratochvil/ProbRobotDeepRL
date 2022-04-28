# ProbRobotDeepRL
*Deep reinforcement learning project for our Probabilistic Robotics class at Tufts.*  
Authors: Niko Ciminelli, Bharat Kesari, Zachary Kratochvil

## About  
This is code for our simulation of a ball-finding task with a Turtlebot.

## Configuring Environment
` pip install poetry ` or ` conda install poetry `  
` poetry install `

## Running Simulation from Pre-trained Model
Not implemented yet.

## Training
To replicate the submitted video, run:  
` poetry run python main.py --num-envs 1 --train --seed 1 --total-timesteps 500 --num-steps 50 --gui `

## Evaluating
To generate the loss curve and other plots after a round of training, run:  
` tensorboard --logdir runs `

## Additional Information
Note: We only use the simulator to
simulate optics. Our observation space is images and PyBullet
renders them nicely. We're using discrete position-control
actions in our robot so we best simulate this by teleporting.

For latest version: https://github.com/zacharykratochvil/ProbRobotDeepRL

## Built using:
* https://github.com/GerardMaggiolino/TRPO-Implementation
* https://github.com/erwincoumans/pybullet_robots
* https://github.com/ericyangyu/PPO-for-Beginners
* https://arxiv.org/abs/1707.06347
