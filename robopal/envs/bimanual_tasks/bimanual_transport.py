import numpy as np

from robopal.envs import BimanualManipulate
import robopal.commons.transform as T
from robopal.wrappers import PettingStyleWrapper

class BimanualTransport(BimanualManipulate):
    name = 'BimanualTransport-v0'
    
    def __init__(self,
                 robot="DualPandaTransport",
                 render_mode='human',
                 control_freq=20,
                 is_show_camera_in_cv=False,
                 controller='CARTIK',
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            is_show_camera_in_cv=is_show_camera_in_cv,
            controller=controller,
        )

        self.obs_dim = (29,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 500

        self.pos_max_bound = {self.agents[0]: np.array([0.7, 0.2, 0.6]),
                              self.agents[1]: np.array([0.7, 0.2, 0.6])}
        self.pos_min_bound = {self.agents[0]: np.array([0.2, -0.2, 0.1]),
                              self.agents[1]: np.array([0.2, -0.2, 0.1])}

    def _get_obs(self, agent) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        # agent1 observations
        obs[0:8] = np.concatenate([
            # gripper position in global coordinates
            a1_end_pos := self.get_site_pos(f'{self.agents[0][-1]}_grip_site'),
            # gripper linear velocity
            self.get_site_xvelp(f'{self.agents[0][-1]}_grip_site') * self.dt,
            self.robot.end[self.agents[0]].get_finger_observations(),
        ], axis=0)

        # agent2 observations
        obs[8:16] = np.concatenate([
            # gripper position in global coordinates
            a2_end_pos := self.get_site_pos(f'{self.agents[1][-1]}_grip_site'),
            # gripper linear velocity
            self.get_site_xvelp(f'{self.agents[1][-1]}_grip_site') * self.dt,
            self.robot.end[self.agents[1]].get_finger_observations(),
        ], axis=0)

        # environment observations
        obs[16:29] = np.concatenate([
            # hammer position
            hammer_pos := self.get_body_pos('hammer'),
            hammer_quat := self.get_body_quat('hammer'),
            # relative position between the hammer and the gripper
            a1_end_pos - hammer_pos,
            a2_end_pos - hammer_pos,
        ], axis=0)

        return obs.copy()
    
    def compute_rewards(self, agent: str):
        dist = self.goal_distance(
            self.get_body_pos('hammer'), 
            self.get_body_pos('carton')
        )
        dist_reward = 1.0 / (1.0 + dist**2)
        dist_reward *= dist_reward
        reward = np.where(dist <= 0.02, dist_reward * 2, dist_reward)
        return reward

    def _get_info(self, agent) -> dict:
        return {
            'is_success': self._is_success(self.get_body_pos('hammer'), self.get_body_pos('carton'), th=0.04)
        }

    def reset_object(self):
        # express the position of the block in the world frame
        # goal_pos_0 = np.random.uniform([0.55, -0.15, 0.46], [0.75, 0.15, 0.66])
        # self.set_site_pos('goal_site0', goal_pos_0)
        pass


if __name__ == "__main__":
    env = BimanualTransport(render_mode='human')
    env = PettingStyleWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        s_, r, terminated, truncated, info = env.step(actions)
        if truncated[env.agents[0]] or truncated[env.agents[1]]:
            env.reset()
    env.close()
