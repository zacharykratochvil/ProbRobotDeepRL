# ProbRobotDeepRL
*Deep reinforcement learning project for our Probabilistic Robotics class at Tufts.*  
Authors: Niko Ciminelli*, Bharat Kesari*, Zachary Kratochvil*
*All authors contributed equally to this work.

## About  
This is code for our simulation of a ball-finding task with a Turtlebot.

## Configuring Environment
First try using poetry. This works especially well if you're
not going to use a GPU.  
` pip install poetry ` or ` conda install poetry `  
then  
` poetry install `  
  
If that fails, try using conda and pip, which works with the A100 GPU with Python 3.9.0.  
` conda install --file requirements_conda.txt -c pytorch `  
then  
` pip install -r requirements_pip.txt `

## Running Simulation from Pre-trained Model
Models are over 200MB each, so we do not provide them here, but we do provide a video called "learned_complex.mp4" If you do train a new model and wish to run it, run:
(with poetry)  
` poetry run python main.py --num-envs 1 --seed 1 --num-minibatches 1 --total-timesteps 500 --num-steps 50 --gui --checkpoint-actor [file]`  
(without poetry)  
` python main.py --num-envs 1 --seed 1 --num-minibatches 1 --total-timesteps 500 --num-steps 50 --gui --checkpoint-actor [file]`  
either command must be followed by the actual path to the checkpoint file of the model you which you wish to load.

## Training
To train, run:  
(with poetry)  
` poetry run python main.py --seed 1 --num-envs 1 --num-steps 300 --num-minibatches 3 --update-epochs 3 --total-timesteps 300000 --train --gym-id TurtleRLEnv-v0 --learning-rate 3e-4 --reward_scheme dense --gui `  
(without poetry)  
` poetry run python main.py --seed 1 --num-envs 1 --num-steps 300 --num-minibatches 3 --update-epochs 3 --total-timesteps 300000 --train --gym-id TurtleRLEnv-v0 --learning-rate 3e-4 --reward_scheme dense --gui `

## Transfer learning
To perform transfer learning, use the arguments:  
` checkpoint-model `, ` --frozen-layers 0 3 `, and ` --zeroed-layers 7 9 11 `  
for example to freeze the convolutional layers and re-initialize the fully connected ones.

## Evaluating
To generate the loss curve and other plots after a round of training, run:  
` tensorboard --logdir runs `  
note you may need to add ` --bind_all ` to run on a remote machine.

## Additional Information
Note: We only use the simulator to
simulate optics. Our observation space is images and PyBullet
renders them nicely. We're using discrete position-control
actions in our robot so we best simulate this by teleporting.

For latest version: https://github.com/zacharykratochvil/ProbRobotDeepRL

## Built using:
* https://medium.com/@gerardmaggiolino/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
* https://github.com/GerardMaggiolino/TRPO-Implementation
* https://github.com/erwincoumans/pybullet_robots
* https://github.com/ericyangyu/PPO-for-Beginners
* https://arxiv.org/abs/1707.06347
* https://www.youtube.com/watch?v=MEt6rrxH8W4
