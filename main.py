import gym
import environment

if __name__ == "__main__":
    env = gym.make("TurtleRLENV-v0", env_type="gui")
    env.reset()