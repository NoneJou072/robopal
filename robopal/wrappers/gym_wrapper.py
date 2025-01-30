import numpy as np
import gymnasium as gym
from gymnasium import spaces
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

    def render(self, mode=None):
        return self.env.render(mode)

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
        obs, reward, terminated, truncated, info = super().step(action)
        obs = {
            'observation': obs.copy(),
            'achieved_goal': self.env._get_achieved_goal(),
            'desired_goal': self.env._get_desired_goal()
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs = {
            'observation': obs.copy(),
            'achieved_goal': self.env._get_achieved_goal(),
            'desired_goal': self.env._get_desired_goal()
        }
        return obs, info

    def compute_reward(self, achieved_goal, desired_goal, info: dict = None, **kwargs):
        return np.vstack([self.env.compute_rewards(achieved_goal, desired_goal, info, **kwargs)] * achieved_goal.shape[0])
