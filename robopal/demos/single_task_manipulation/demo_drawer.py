import numpy as np

from robopal.envs import ManipulateEnv
from robopal.robots.diana_med import DianaDrawer


class DrawerEnv(ManipulateEnv):
    """
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=DianaDrawer(),
                 render_mode='human',
                 control_freq=10,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 is_pd=False,
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
            is_interpolate=is_interpolate,
            is_pd=is_pd,
        )
        self.name = 'Drawer-v1'

        self.obs_dim = (17,)
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.pos_max_bound = np.array([0.65, 0.2, 0.4])
        self.pos_min_bound = np.array([0.3, -0.2, 0.14])

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

        return {
            'observation': obs.copy(),
            'achieved_goal': object_pos.copy(),  # handle position
            'desired_goal': self.get_site_pos('drawer_goal').copy()
        }

    def _get_info(self) -> dict:
        return {'is_success': self._is_success(self.get_site_pos('drawer'), self.get_site_pos('drawer_goal'), th=0.02)}

    def reset_object(self):
        goal_pos = np.array([0.0, np.random.uniform(-0.2, -0.1), 0.05])
        site_id = self.get_site_id('drawer_goal')
        self.mj_model.site_pos[site_id] = goal_pos


if __name__ == "__main__":
    env = DrawerEnv()
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, _ = env.step(action)
        if truncated:
            env.reset()
    env.close()
