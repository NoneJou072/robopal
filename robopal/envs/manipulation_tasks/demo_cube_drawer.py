import numpy as np

from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaDrawerCube
from robopal.wrappers import GoalEnvWrapper


class DrawerCubeEnv(ManipulateEnv):

    name = 'DrawerBox-v1'
    
    def __init__(self,
                 robot=DianaDrawerCube,
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

        self.obs_dim = (35,)
        self.goal_dim = (9,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.TASK_FLAG = 0

        self.pos_max_bound = np.array([0.6, 0.2, 0.37])
        self.pos_min_bound = np.array([0.3, -0.2, 0.12])
        self.grip_max_bound = 0.02
        self.grip_min_bound = -0.01

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        # gripper state
        obs[0:8] = np.concatenate((
            # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site'),
            # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt,
            self.mj_data.joint('0_r_finger_joint').qpos,
            self.mj_data.joint('0_r_finger_joint').qvel * self.dt
        ))

        if self.TASK_FLAG == 0:
            obs[11:14] = (  # handle position in global coordinates
                handle_pos := self.get_site_pos('drawer')
            )
            obs[14:17] = end_pos - handle_pos  # distance between the handle and the end
            # velocity with respect to the gripper
            handle_velp = self.get_site_xvelp('drawer') * self.dt
            obs[17:20] = (  # velocity with respect to the gripper
                handle_velp - end_vel
            )

        # drawer state
        if self.TASK_FLAG == 1:
            obs[20:23] = (  # block position in global coordinates
                block_pos := self.get_body_pos('green_block')
            )
            obs[23:26] = end_pos - block_pos  # distance between the block and the end
            obs[26:29] = (  # block rotation
                trans.mat_2_euler(self.get_body_rotm('green_block'))
            )
            block_velp = self.get_body_xvelp('green_block') * self.dt
            obs[29:32] = (  # velocity with respect to the gripper
                block_velp # - end_vel
            )

        return obs.copy()

    def _get_achieved_goal(self):
        achieved_goal = np.concatenate([
            self.get_site_pos('0_grip_site'),
            self.get_site_pos('drawer'),
            self.get_body_pos('green_block')
        ], axis=0)
        return achieved_goal.copy()

    def _get_desired_goal(self):
        if self._is_success(self.get_site_pos('drawer'), self.get_site_pos('drawer_goal')) == 0:
            reach_goal = self.get_site_pos('drawer')
        else:
            reach_goal = self.get_body_pos('green_block')

        desired_goal = np.concatenate([
            reach_goal,
            self.get_site_pos('drawer_goal'),
            self.get_site_pos('cube_goal'),
        ], axis=0)
        return desired_goal.copy()

    def _get_info(self) -> dict:
        return {
            'is_drawer_success': self._is_success(self.get_site_pos('drawer'), self.get_site_pos('drawer_goal')),
            'is_place_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('cube_goal'))
        }

    def reset_object(self):
        if self.TASK_FLAG == 1:
            self.mj_data.joint('drawer:joint').qpos[0] = 0.14
        # reset object position
        random_x_pos = np.random.uniform(0.35, 0.4)
        random_y_pos = np.random.uniform(-0.15, 0.15)
        self.set_object_pose('green_block:joint', np.array([random_x_pos, random_y_pos, 0.44, 1.0, 0.0, 0.0, 0.0]))
        self.set_site_pos('cube_goal', np.array([0.59, 0.0, 0.478]))

        return super().reset_object()


if __name__ == "__main__":
    env = DrawerCubeEnv()
    env = GoalEnvWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
