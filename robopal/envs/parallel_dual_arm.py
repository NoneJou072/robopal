import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from robopal.envs import RobotEnv
from robopal.robots.diana_med import DualDianaMed


class BimanualPettingStyleEnv(ParallelEnv, RobotEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "dual_arm_v0",
        "render_modes": [
            "human",
            "rgb_array",
            "depth",
            "unity",
        ],
    }

    def __init__(self):
        """ The init method takes in environment arguments.
        """
        super().__init__(
            robot=DualDianaMed,
            render_mode="human",
            control_freq=10,
            enable_camera_viewer=False,
            controller="JNTIMP",
            is_interpolate=False,
        )

        self.obs_dim = (35,)
        self.action_dim = (3,)

        self.timestep = None
        self.possible_agents = self.robot.agents

    def reset(self, seed=None, options=None):
        """ Reset set the environment to a starting point.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        observations = {
            a: np.zeros(self.obs_dim, dtype=np.float32)
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        """ Takes in an action for the current agent (specified by agent_selection).
        """
        # Execute actions
        RobotEnv.step(self, actions)

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1

        # Get observations
        observations = {
            a: np.zeros(self.obs_dim, dtype=np.float32)
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """ Renders the environment.
        """
        RobotEnv.render(self)

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Box(low=-np.inf, high=np.inf, shape=self.obs_dim, dtype="float64")

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=self.action_dim, dtype="float64")
