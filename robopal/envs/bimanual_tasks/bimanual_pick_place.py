import numpy as np

import robopal
from robopal.envs import BimanualManipulate
import robopal.commons.transform as trans
from robopal.wrappers import PettingStyleWrapper


class BimanualPickAndPlace(BimanualManipulate):
    name = 'BimanualPickAndPlace-v0'
    
    def __init__(self,
                 robot='DualPandaPickAndPlace',
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

        self.obs_dim = (21,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.pos_max_bound = {self.agents[0]: np.array([0.65, 0.2, 0.6]),
                              self.agents[1]: np.array([0.65, 0.2, 0.6])}
        self.pos_min_bound = {self.agents[0]: np.array([0.3, -0.2, 0.2]),
                              self.agents[1]: np.array([0.3, -0.2, 0.2])}

    def _get_obs(self, agent: str = None) -> np.ndarray:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[0:8] = np.concatenate([
            # gripper position in global coordinates
            end_pos := self.get_site_pos(f'{agent[-1]}_grip_site'),
            # gripper linear velocity
            end_vel := self.get_site_xvelp(f'{agent[-1]}_grip_site') * self.dt,
            self.robot.end[agent].get_finger_observations(),
        ], axis=0)

        obs[8:21] = np.concatenate([
            # block position in global coordinates
            object_pos := self.get_body_pos('green_block'),
            # Relative block position with respect to gripper position in globla coordinates.
            end_pos - object_pos,
            # block rotation
            trans.mat_2_quat(self.get_body_rotm('green_block')),
            # block linear velocity
            self.get_body_xvelp('green_block') * self.dt
        ], axis=0)

        return obs.copy()

    def _get_info(self, agent: str = None) -> dict:
        return {
            'is_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('goal_site'), th=0.02)
        }

    def reset_object(self):
        random_x_pos = np.random.uniform(0.35, 0.55)
        random_y_pos = np.random.uniform(-0.15, 0.15)
        self.set_object_pose('green_block:joint', np.array([random_x_pos, random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        goal_pos = np.random.uniform([0.35, -0.15, 0.46], [0.55, 0.15, 0.66])
        block_pos = np.array([random_x_pos, random_y_pos, 0.46])
        while np.linalg.norm(block_pos - goal_pos) <= 0.05:
            goal_pos = np.random.uniform([0.35, -0.15, 0.46], [0.55, 0.15, 0.66])
        self.set_site_pos('goal_site', goal_pos)


if __name__ == "__main__":
    env = robopal.make(
        'BimanualPickAndPlace-v0',
        robot='DualPandaPickAndPlace',
        render_mode='human'
    )
    env = PettingStyleWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        actions = {
            agent: np.random.uniform(env.min_action, env.max_action, env.action_dim)
            for agent in env.agents
        }
        s_, r, terminated, truncated, info = env.step(actions)
        if truncated[env.agents[0]] or truncated[env.agents[1]]:
            env.reset()
    env.close()
