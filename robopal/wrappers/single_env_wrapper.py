import numpy as np
from robopal.envs.bimanual_tasks.bimanual_manipulate import BimanualManipulate


class SingleEnvWrapper:
    """ Convert a parallel multi-agents environment to a single-agent environment

    :param pallel_env: The parallel multi-agents environment
    :param mode: The mode of the single-agent environment, possible values are 'combined' and 'separate'
    """
    def __init__(self, pallel_env: BimanualManipulate, mode="combined") -> None:
        self._parallel_env = pallel_env
        self.mode = mode
        assert mode in ['combined', 'shared'], "The mode should be 'combined' or 'shared'"
        self.action_dim = (len(self._parallel_env.agents) * self._parallel_env.action_dim[0],)
        if mode == 'combined':
            self.obs_dim = (len(self._parallel_env.agents) * self._parallel_env.obs_dim[0],)
        else:
            self.obs_dim = self._parallel_env.obs_dim

    def step(self, action: np.ndarray):
        # convert array type action to dict type action
        actions = {agent: action[i*self._parallel_env.action_dim[0]:(i+1)*self._parallel_env.action_dim[0]]
                   for i, agent in enumerate(self._parallel_env.agents)}

        observations, rewards, terminations, truncations, infos = self._parallel_env.step(actions)

        if self.mode == 'combined':
            return (
                np.concatenate([observations[agent] for agent in self._parallel_env.agents]),
                np.sum(list(rewards.values())),
                all(terminations.values()),
                all(truncations.values()),
                {agent: infos[agent] for agent in self._parallel_env.agents}
            )
        elif self.mode == 'shared':  # only return the information of the first agent
            return (
                observations[self._parallel_env.agents[0]],
                rewards[self._parallel_env.agents[0]],
                terminations[self._parallel_env.agents[0]],
                truncations[self._parallel_env.agents[0]],
                infos[self._parallel_env.agents[0]]
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def reset(self, **kwargs):
        observations, infos = self._parallel_env.reset(**kwargs)
        return (
            observations[self._parallel_env.agents[0]],
            infos[self._parallel_env.agents[0]]
        )

    def close(self):
        self._parallel_env.close()

    def render(self, mode='human'):
        self._parallel_env.render(mode)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self._parallel_env, attr)
        else:
            return getattr(self, attr)

    @property
    def unwrapped(self):
        return self._parallel_env
