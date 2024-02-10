import numpy as np

from robopal.envs import ManipulateEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaGraspMultiObjs


class MultiCubes(ManipulateEnv):
    """
    The control frequency of the robot is of f = 10 Hz. This is achieved by applying the same action
    in 100 subsequent simulator step (with a time step of dt = 0.001 s) before returning the control to the robot.
    """

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

    def action_scale(self, action):
        pos_offset = 0.1 * action[:3]
        actual_pos_action = self.kdl_solver.fk(self.robot.arm_qpos)[0] + pos_offset

        pos_max_bound = np.array([0.68, 0.25, 0.28])
        pos_min_bound = np.array([0.3, -0.25, 0.13])
        actual_pos_action = actual_pos_action.clip(pos_min_bound, pos_max_bound)

        # Map to target action space bounds
        grip_max_bound = 0.02
        grip_min_bound = -0.01
        gripper_ctrl = (action[3] + 1) * (grip_max_bound - grip_min_bound) / 2 + grip_min_bound
        return actual_pos_action, gripper_ctrl

    def step(self, action) -> tuple:
        return super().step(action)

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        The actual observation is format the table below.
        | gripper position | blocks position & rotation | gripper vel | gripper_qpos | gripper_qvel |
        """
        obs = np.zeros(self.obs_dim)

        obs[0:3] = (  # gripper position
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # block position
            red_block_pos := self.get_body_pos('red_block')
        )
        obs[6:9] = (  # block position
            green_block_pos := self.get_body_pos('green_block')
        )
        obs[9:12] = (  # block position
            blue_block_pos := self.get_body_pos('blue_block')
        )
        obs[12:15] = (  # distance between the block and the end
            end_pos - red_block_pos
        )
        obs[15:18] = (  # distance between the block and the end
            end_pos - green_block_pos
        )
        obs[18:21] = (  # distance between the block and the end
            end_pos - blue_block_pos
        )
        obs[21:24] = (  # block rotation
            trans.mat_2_euler(self.get_body_rotm('red_block'))
        )
        obs[24:27] = (  # block rotation
            trans.mat_2_euler(self.get_body_rotm('green_block'))
        )
        obs[27:30] = (  # block rotation
            trans.mat_2_euler(self.get_body_rotm('blue_block'))
        )
        obs[30:33] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt
        )
        red_block_velp = self.get_body_xvelp('red_block') * self.dt
        green_block_velp = self.get_body_xvelp('green_block') * self.dt
        blue_block_velp = self.get_body_xvelp('blue_block') * self.dt
        obs[33:36] = (  # velocity with respect to the gripper
            red_block_velp - end_vel
        )
        obs[36:39] = (  # velocity with respect to the gripper
            green_block_velp - end_vel
        )
        obs[39:42] = (  # velocity with respect to the gripper
            blue_block_velp - end_vel
        )
        # obs[36:39] = self.get_body_xvelr('green_block') * self.dt
        obs[42] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[43] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt

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
            reach_goal = self.get_body_pos('red_block')
        elif self._is_success(self.get_body_pos('green_block'), self.get_site_pos('green_goal'), th=0.02) == 0:
            reach_goal = self.get_body_pos('green_block')
        else:
            reach_goal = self.get_body_pos('blue_block')

        desired_goal = np.concatenate([
            reach_goal,
            self.get_site_pos('red_goal'),
            self.get_site_pos('green_goal'),
            self.get_site_pos('blue_goal'),
        ], axis=0)
        return desired_goal.copy()

    def _get_info(self) -> dict:
        return {
            'is_red_success': self._is_success(self.get_body_pos('red_block'), self.get_site_pos('red_goal'), th=0.02),
            'is_green_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('green_goal'), th=0.02),
            'is_blue_success': self._is_success(self.get_body_pos('blue_block'), self.get_site_pos('blue_goal'), th=0.02)
        }

    def reset(self, seed=None):
        return super().reset()

    def reset_object(self):
        r_random_x_pos = np.random.uniform(0.4, 0.55)
        r_random_y_pos = np.random.uniform(-0.15, 0.15)
        self.set_object_pose('red_block:joint', np.array([r_random_x_pos, r_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        g_random_x_pos = np.random.uniform(0.4, 0.55)
        g_random_y_pos = np.random.uniform(-0.15, 0.15)
        while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([g_random_x_pos, g_random_y_pos])) < 0.08:
            g_random_x_pos = np.random.uniform(0.4, 0.55)
            g_random_y_pos = np.random.uniform(-0.15, 0.15)
        self.set_object_pose('green_block:joint', np.array([g_random_x_pos, g_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        b_random_x_pos = np.random.uniform(0.4, 0.55)
        b_random_y_pos = np.random.uniform(-0.15, 0.15)
        while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([b_random_x_pos, b_random_y_pos])) < 0.08 \
            or np.linalg.norm(np.array([g_random_x_pos, g_random_y_pos]) - np.array([b_random_x_pos, b_random_y_pos])) < 0.08:
            b_random_x_pos = np.random.uniform(0.4, 0.55)
            b_random_y_pos = np.random.uniform(-0.15, 0.15)
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
    for t in range(int(1e6)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
