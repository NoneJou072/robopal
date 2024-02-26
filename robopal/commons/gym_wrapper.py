import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GymWrapper(gym.Env):
    def __init__(self, env) -> None:
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

    def __init__(self, env) -> None:
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

    def compute_reward(self, achieved_goal, desired_goal, **kwargs):
        return self.env.compute_rewards(achieved_goal, desired_goal, **kwargs)
