import numpy as np

from robopal.demos.bimanual_tasks import BimanualManipulate
import robopal.commons.transform as trans
from robopal.robots.dual_arms import DualDianaReach
from robopal.commons.wrappers import PettingStyleWrapper

class BimanualReach(BimanualManipulate):

    def __init__(self,
                 robot=DualDianaReach,
                 render_mode='human',
                 control_freq=20,
                 enable_camera_viewer=False,
                 controller='CARTIK',
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
        )
        self.name = 'BimanualReach-v0'

        self.obs_dim = (14,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.pos_max_bound = {self.agents[0]: np.array([0.65, 0.2, 0.6]),
                              self.agents[1]: np.array([0.65, 0.2, 0.6])}
        self.pos_min_bound = {self.agents[0]: np.array([0.3, -0.2, 0.2]),
                              self.agents[1]: np.array([0.3, -0.2, 0.2])}

    def _get_obs(self, agent) -> dict:
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
            self.mj_data.joint(f'{agent[-1]}_r_finger_joint').qpos,
            self.mj_data.joint(f'{agent[-1]}_r_finger_joint').qvel * self.dt,
        ], axis=0)
        obs[8:14] = np.concatenate([
            # goal position in global coordinates
            goal_pos := self.get_site_pos(f'goal_site{agent[-1]}'),
            # relative position between the site and the gripper
            end_pos - goal_pos,
        ], axis=0)

        return obs.copy()
    
    def compute_rewards(self, agent: str):
        dist = self.goal_distance(
            self.get_site_pos(f'{agent[-1]}_grip_site'), 
            self.get_site_pos(f'goal_site{agent[-1]}')
        )
        dist_reward = 1.0 / (1.0 + dist**2)
        dist_reward *= dist_reward
        reward = np.where(dist <= 0.02, dist_reward * 2, dist_reward)
        return reward

    def _get_info(self, agent) -> dict:
        return {}
        # return {'is_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('goal_site'), th=0.02)}

    def reset_object(self):
        # express the position of the block in the world frame
        # goal1 -> agent 0, goal2 -> agent 1
        goal1_x_pos = np.random.uniform(0.55, 0.75)
        goal1_y_pos = np.random.uniform(-0.15, 0.15)
        goal1_z_pos = np.random.uniform(0.46, 0.66)
        goal1_pos = np.array([goal1_x_pos, goal1_y_pos, goal1_z_pos])
        self.set_site_pose('goal_site0', goal1_pos)

        goal2_x_pos = np.random.uniform(0.15, 0.35)
        goal2_y_pos = np.random.uniform(-0.15, 0.15)
        goal2_z_pos = np.random.uniform(0.46, 0.66)
        goal2_pos = np.array([goal2_x_pos, goal2_y_pos, goal2_z_pos])
        while np.linalg.norm(goal1_pos - goal2_pos) <= 0.05:
            goal2_x_pos = np.random.uniform(0.15, 0.35)
            goal2_y_pos = np.random.uniform(-0.15, 0.15)
            goal2_z_pos = np.random.uniform(0.45, 0.66)
            goal2_pos = np.array([goal2_x_pos, goal2_y_pos, goal2_z_pos])
        self.set_site_pose('goal_site1', goal2_pos)


if __name__ == "__main__":
    env = BimanualReach(render_mode='human')
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
