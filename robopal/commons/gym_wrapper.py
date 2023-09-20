import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GymWrapper(gym.Env):
    def __init__(self, env) -> None:
        super(GymWrapper, self).__init__()
        self.env = env

        self.observation_space = spaces.Dict(
            {
                'observations': spaces.Box(low=-np.inf, high=np.inf, shape=self.env.obs_dim),
            }
        )
        self.action_space = spaces.Box(
            low = env.min_action, high = env.max_action, shape = self.env.action_dim
        )

        self.max_episode_steps = env.max_episode_steps
        self.name = env.name

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed)

    def render(self, mode="human"):
        self.env.render()
