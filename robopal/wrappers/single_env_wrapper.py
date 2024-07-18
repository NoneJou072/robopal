import numpy as np
from robopal.envs.bimanual_tasks.bimanual_manipulate import BimanualManipulate


class SingleEnvWrapper:
    """ Convert a parallel multi-agents environment to a single-agent environment
    """
    def __init__(self, pallel_env: BimanualManipulate) -> None:
        self.parallel_env = pallel_env

    def step(self, action: np.ndarray):
        # convert array type action to dict type action
        actions = {
            self.parallel_env.agents[0]: action,
            self.parallel_env.agents[1]: action
        }

        observations, rewards, terminations, truncations, infos = self.parallel_env.step(actions)

        return (
            observations[self.parallel_env.agents[0]],
            rewards[self.parallel_env.agents[0]],
            terminations[self.parallel_env.agents[0]],
            truncations[self.parallel_env.agents[0]],
            infos[self.parallel_env.agents[0]]
        )

    def reset(self, **kwargs):
        observations, infos = self.parallel_env.reset(**kwargs)
        return (
            observations[self.parallel_env.agents[0]],
            infos[self.parallel_env.agents[0]]
        )

    def close(self):
        self.parallel_env.close()

    def render(self, mode='human'):
        self.parallel_env.render(mode)
