import functools
from copy import copy
import logging
from typing import Dict

import numpy as np
from gymnasium import spaces
try:
    from pettingzoo import ParallelEnv
except ImportError:
    logging.warning("PettingZoo not installed. PettingStyleWrapper will not be available.")

from robopal.envs import MujocoEnv


class PettingStyleWrapper(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth",
            "unity",
        ],
    }

    def __init__(self, env: MujocoEnv):
        """ The init method takes in environment arguments.
        """

        self.env = env
        self.metadata['name'] = self.env.name
        self.possible_agents = self.env.robot.agents

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset(self, seed=None, options=None):
        """ Reset set the environment to a starting point.
        """
        self.agents = copy(self.possible_agents)

        return self.env.reset(seed=seed, options=options)

    def step(self, actions: Dict[str, np.ndarray]):
        """ Takes in an action for the current agent (specified by agent_selection).
        """
        return self.env.step(actions)

    def render(self, mode=None):
        """ Renders the environment.
        """
        return self.env.render(self, mode)

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Box(low=-np.inf, high=np.inf, shape=self.env.obs_dim, dtype="float64")

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=self.env.action_dim, dtype="float64")
