import numpy as np

from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
from robopal.robots.diana_med import DianaDrawer
from robopal.wrappers import GoalEnvWrapper


class DrawerEnv(ManipulateEnv):

    name = 'Drawer-v1'
    
    def __init__(self,
                 robot=DianaDrawer,
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

        self.obs_dim = (17,)
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[:3] = (  # gripper position
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # drawer position
            object_pos := self.get_site_pos('drawer')
        )
        obs[6:9] = (  # distance between the block and the end
            object_rel_pos := end_pos - object_pos
        )
        obs[9:12] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt
        )
        object_velp = self.get_site_xvelp('drawer') * self.dt
        obs[12:15] = (  # velocity with respect to the gripper
            object2end_velp := object_velp - end_vel
        )
        obs[15] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[16] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt

        return obs.copy()
    
    def _get_achieved_goal(self) -> np.ndarray:
        return self.get_site_pos('drawer')

    def _get_desired_goal(self) -> np.ndarray:
        return self.get_site_pos('drawer_goal')

    def _get_info(self) -> dict:
        return {'is_success': self._is_success(self.get_site_pos('drawer'), self.get_site_pos('drawer_goal'), th=0.02)}

    def reset_object(self):
        goal_pos = np.array([0.0, np.random.uniform(-0.2, -0.1), 0.05])
        site_id = self.get_site_id('drawer_goal')
        self.mj_model.site_pos[site_id] = goal_pos

        return super().reset_object()


if __name__ == "__main__":
    env = DrawerEnv()
    env = GoalEnvWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, _ = env.step(action)
        if truncated:
            env.reset()
    env.close()
