import functools
from copy import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces
try:
    from pettingzoo import ParallelEnv
except:
    pass
from robopal.envs import RobotEnv


class GymWrapper(gym.Env):
    def __init__(self, env: RobotEnv) -> None:
        super(GymWrapper, self).__init__()
        self.env = env

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.env.obs_dim, dtype="float64"
        )

        self.action_space = spaces.Box(
            low=env.min_action, high=env.max_action, shape=self.env.action_dim, dtype="float64"
        )

        self.max_episode_steps = env.max_episode_steps
        self.name = env.name
        self.render_mode = env.render_mode

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        # following line to seed self.np_random
        super().reset(seed=seed, options=options)

        return self.env.reset(seed=seed, options=options)

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()


class GoalEnvWrapper(GymWrapper):
    """ GoalEnvWrapper: a wrapper for gym.GoalEnv """

    def __init__(self, env: RobotEnv) -> None:
        super(GoalEnvWrapper, self).__init__(env)
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low=-np.inf, high=np.inf, shape=self.env.obs_dim, dtype="float64"),
                desired_goal=spaces.Box(low=-np.inf, high=np.inf, shape=self.env.goal_dim, dtype="float64"),
                achieved_goal=spaces.Box(low=-np.inf, high=np.inf, shape=self.env.goal_dim, dtype="float64"),
            )
        )

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def compute_reward(self, achieved_goal, desired_goal, info: dict = None, **kwargs):
        return self.env.compute_rewards(achieved_goal, desired_goal, info, **kwargs)


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

    def __init__(self, env: RobotEnv):
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

    def step(self, actions: dict[str, np.ndarray]):
        """ Takes in an action for the current agent (specified by agent_selection).
        """
        return self.env.step(actions)

    def render(self):
        """ Renders the environment.
        """
        self.env.render(self)

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
        return spaces.Box(low=-np.inf, high=np.inf, shape=self.env.action_dim, dtype="float64")
