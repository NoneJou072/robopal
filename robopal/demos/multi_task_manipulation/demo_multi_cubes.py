import numpy as np

from robopal.envs import ManipulateEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaGraspMultiObjs


class MultiCubes(ManipulateEnv):

    def __init__(self,
                 robot=DianaGraspMultiObjs(),
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
        self.name = 'MultiCubeStack-v2'

        self.obs_dim = (44,)
        self.goal_dim = (12,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.TASK_FLAG = 0

        self.pos_max_bound = np.array([0.68, 0.25, 0.28])
        self.pos_min_bound = np.array([0.3, -0.25, 0.13])
        self.grip_max_bound = 0.02
        self.grip_min_bound = -0.02

        self.red_init_pos = None
        self.green_init_pos = None
        self.blue_init_pos = None

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        The actual observation is format the table below.
        | gripper position | blocks position & rotation | gripper vel | gripper_qpos | gripper_qvel |
        """
        obs = np.zeros(self.obs_dim)

        # gripper state
        obs[0:3] = (  # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt
        )
        obs[6] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[7] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt

        # red block state
        if self.TASK_FLAG == 0:
            obs[8:11] = (  # block position
                red_block_pos := self.get_body_pos('red_block')
            )
            obs[11:14] = (  # distance between the block and the end
                end_pos - red_block_pos
            )
            obs[14:17] = (  # block rotation
                trans.mat_2_euler(self.get_body_rotm('red_block'))
            )
            red_block_velp = self.get_body_xvelp('red_block') * self.dt
            obs[17:20] = (  # velocity with respect to the gripper
                red_block_velp - end_vel
            )

        # green block state
        if self.TASK_FLAG == 1:
            obs[20:23] = (  # block position
                green_block_pos := self.get_body_pos('green_block')
            )
            obs[23:26] = (  # distance between the block and the end
                end_pos - green_block_pos
            )
            obs[26:29] = (  # block rotation
                trans.mat_2_euler(self.get_body_rotm('green_block'))
            )
            green_block_velp = self.get_body_xvelp('green_block') * self.dt
            obs[29:32] = (  # velocity with respect to the gripper
                green_block_velp - end_vel
            )

        # blue block state
        if self.TASK_FLAG == 2:
            obs[32:35] = (  # block position
                blue_block_pos := self.get_body_pos('blue_block')
            )
            obs[35:38] = (  # distance between the block and the end
                end_pos - blue_block_pos
            )
            obs[38:41] = (  # block rotation
                trans.mat_2_euler(self.get_body_rotm('blue_block'))
            )
            blue_block_velp = self.get_body_xvelp('blue_block') * self.dt
            obs[41:44] = (  # velocity with respect to the gripper
                blue_block_velp - end_vel
            )

        return {
            'observation': obs.copy(),
            'achieved_goal': self._get_achieved_goal(),
            'desired_goal': self._get_desired_goal()
        }

    def _get_achieved_goal(self):
        achieved_goal = np.concatenate([
            self.get_site_pos('0_grip_site'),
            self.get_body_pos('red_block'),
            self.get_body_pos('green_block'),
            self.get_body_pos('blue_block'),
        ], axis=0)
        return achieved_goal.copy()

    def _get_desired_goal(self):
        if self._is_success(self.get_body_pos('red_block'), self.get_site_pos('red_goal'), th=0.02) == 0:
            return np.concatenate([
                self.get_body_pos('red_block'),
                self.get_site_pos('red_goal'),
                self.green_init_pos,
                self.blue_init_pos,
            ], axis=0).copy()
        elif self._is_success(self.get_body_pos('green_block'), self.get_site_pos('green_goal'), th=0.02) == 0:
            return np.concatenate([
                self.get_body_pos('green_block'),
                self.get_site_pos('red_goal'),
                self.get_site_pos('green_goal'),
                self.blue_init_pos,
            ], axis=0).copy()
        else:
            return np.concatenate([
                self.get_body_pos('blue_block'),
                self.get_site_pos('red_goal'),
                self.get_site_pos('green_goal'),
                self.get_site_pos('blue_goal'),
            ], axis=0).copy()

    def _get_info(self) -> dict:
        return {
            'is_red_success': self._is_success(self.get_body_pos('red_block'), self.get_site_pos('red_goal'), th=0.02),
            'is_green_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('green_goal'), th=0.02),
            'is_blue_success': self._is_success(self.get_body_pos('blue_block'), self.get_site_pos('blue_goal'), th=0.02)
        }

    def reset_object(self):
        # set the position of the red, green, and blue blocks
        r_random_x_pos = np.random.uniform(0.4, 0.55)
        r_random_y_pos = np.random.uniform(-0.15, 0.15)
        self.red_init_pos = np.array([r_random_x_pos, r_random_y_pos, 0.46])
        self.set_object_pose('red_block:joint', np.array([r_random_x_pos, r_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        g_random_x_pos = np.random.uniform(0.4, 0.55)
        g_random_y_pos = np.random.uniform(-0.15, 0.15)
        while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([g_random_x_pos, g_random_y_pos])) < 0.08:
            g_random_x_pos = np.random.uniform(0.4, 0.55)
            g_random_y_pos = np.random.uniform(-0.15, 0.15)
        self.green_init_pos = np.array([g_random_x_pos, g_random_y_pos, 0.46])
        self.set_object_pose('green_block:joint', np.array([g_random_x_pos, g_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        b_random_x_pos = np.random.uniform(0.4, 0.55)
        b_random_y_pos = np.random.uniform(-0.15, 0.15)
        while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([b_random_x_pos, b_random_y_pos])) < 0.08 \
            or np.linalg.norm(np.array([g_random_x_pos, g_random_y_pos]) - np.array([b_random_x_pos, b_random_y_pos])) < 0.08:
            b_random_x_pos = np.random.uniform(0.4, 0.55)
            b_random_y_pos = np.random.uniform(-0.15, 0.15)
        self.blue_init_pos = np.array([b_random_x_pos, b_random_y_pos, 0.46])
        self.set_object_pose('blue_block:joint', np.array([b_random_x_pos, b_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        # red goal
        random_goal_x_pos = np.random.uniform(0.4, 0.55)
        random_goal_y_pos = np.random.uniform(-0.15, 0.15)
        while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([random_goal_x_pos, random_goal_y_pos])) < 0.06 \
            or np.linalg.norm(np.array([g_random_x_pos, g_random_y_pos]) - np.array([random_goal_x_pos, random_goal_y_pos])) < 0.06 \
            or np.linalg.norm(np.array([b_random_x_pos, b_random_y_pos]) - np.array([random_goal_x_pos, random_goal_y_pos])) < 0.06:
            random_goal_x_pos = np.random.uniform(0.4, 0.55)
            random_goal_y_pos = np.random.uniform(-0.15, 0.15)
        red_goal = np.array([random_goal_x_pos, random_goal_y_pos, 0.44])
        self.set_site_pose('red_goal', red_goal)

        # green goal
        green_goal = np.array([red_goal[0], red_goal[1], red_goal[2] + 0.04])
        self.set_site_pose('green_goal', green_goal)

        # blue goal
        blue_goal = np.array([green_goal[0], green_goal[1], green_goal[2] + 0.04])
        self.set_site_pose('blue_goal', blue_goal)

        if self.TASK_FLAG == 0:
            pass
        elif self.TASK_FLAG == 1:
            self.set_object_pose('red_block:joint', np.array([red_goal[0], red_goal[1], red_goal[2], 1.0, 0.0, 0.0, 0.0]))
        elif self.TASK_FLAG == 2:
            self.set_object_pose('red_block:joint', np.array([red_goal[0], red_goal[1], red_goal[2], 1.0, 0.0, 0.0, 0.0]))
            self.set_object_pose('green_block:joint', np.array([red_goal[0], red_goal[1], red_goal[2] + 0.04, 1.0, 0.0, 0.0, 0.0]))


if __name__ == "__main__":
    env = MultiCubes()
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
