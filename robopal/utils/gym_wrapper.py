import os
import sys

dir_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.dirname(dir_path))

import numpy as np
import gym
from gym import spaces


class GymWrapper(gym.Env):
    def __init__(self, env) -> None:
        super(GymWrapper, self).__init__()
        self.env = env

        self.action_space = spaces.Box(env.action_low, env.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.env.obs_dim)
        self.max_episode_steps = env.max_episode_steps
        self.name = env.name

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        self.env.render()
