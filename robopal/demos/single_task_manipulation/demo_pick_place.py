import numpy as np
import logging

from robopal.envs.task_ik_ctrl_env import PosCtrlEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaGrasp

logging.basicConfig(level=logging.INFO)


class PickAndPlaceEnv(PosCtrlEnv):
    """ Reference: https://robotics.farama.org/envs/fetch/pick_and_place/#
    The control frequency of the robot is of f = 10 Hz. This is achieved by applying the same action
    in 100 subsequent simulator step (with a time step of dt = 0.001 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=DianaGrasp(),
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
        self.name = 'PickAndPlace-v1'

        self.obs_dim = (23,)
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50
        self._timestep = 0

        self.goal_pos = None

    def action_scale(self, action):
        pos_offset = 0.1 * action[:3]
        actual_pos_action = self.kdl_solver.fk(self.robot.arm_qpos)[0] + pos_offset

        pos_max_bound = np.array([0.6, 0.2, 0.37])
        pos_min_bound = np.array([0.3, -0.2, 0.12])
        actual_pos_action = actual_pos_action.clip(pos_min_bound, pos_max_bound)

        # Map to target action space bounds
        grip_max_bound = 0.02
        grip_min_bound = -0.01
        gripper_ctrl = (action[3] + 1) * (grip_max_bound - grip_min_bound) / 2 + grip_min_bound
        return actual_pos_action, gripper_ctrl

    def step(self, action) -> tuple:
        """ Take one step in the environment.

        :param action:  The action space is 4-dimensional, with the first 3 dimensions corresponding to the desired
        position of the block in Cartesian coordinates, and the last dimension corresponding to the
        desired gripper opening (0 for closed, 1 for open).
        :return: obs, reward, terminated, truncated, info
        """
        self._timestep += 1

        actual_pos_action, gripper_ctrl = self.action_scale(action)
        # take one step
        self.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = gripper_ctrl
        self.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = gripper_ctrl

        super().step(actual_pos_action[:3])

        obs = self._get_obs()
        reward = self.compute_rewards(obs['achieved_goal'], obs['desired_goal'])
        terminated = False
        truncated = True if self._timestep >= self.max_episode_steps else False
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def compute_rewards(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict = None):
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return -(d > 0.05).astype(np.float64)

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[0:3] = (  # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # block position in global coordinates
            object_pos := self.get_body_pos('green_block')
        )
        obs[6:9] = (  # Relative block position with respect to gripper position in globla coordinates.
            object_rel_pos := end_pos - object_pos
        )
        obs[9:12] = (  # block rotation
            trans.mat_2_euler(self.get_body_rotm('green_block'))
        )
        obs[12:15] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt
        )
        object_velp = self.get_body_xvelp('green_block') * self.dt
        obs[15:18] = (  # velocity with respect to the gripper
            object2end_velp := object_velp - end_vel
        )

        obs[18:21] = self.get_body_xvelr('green_block') * self.dt
        obs[21] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[22] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt

        return {
            'observation': obs.copy(),
            'achieved_goal': object_pos.copy(),  # block position
            'desired_goal': self.goal_pos.copy()
        }

    def _get_info(self) -> dict:
        return {'is_success': self._is_success(self.get_body_pos('green_block'), self.goal_pos)}

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        """ Compute whether the achieved goal successfully achieved the desired goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < 0.05).astype(np.float32)

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def reset(self, seed=None):
        super().reset()
        self._timestep = 0
        # set new goal
        self.goal_pos = self.get_site_pos('goal_site')

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, info

    def reset_object(self):
        random_x_pos = np.random.uniform(0.35, 0.55)
        random_y_pos = np.random.uniform(-0.15, 0.15)
        self.set_object_pose('green_block:joint', np.array([random_x_pos, random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        random_goal_x_pos = np.random.uniform(0.35, 0.55)
        random_goal_y_pos = np.random.uniform(-0.15, 0.15)
        random_goal_z_pos = np.random.uniform(0.46, 0.66)

        block_pos = np.array([random_x_pos, random_y_pos, 0.46])
        goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        while np.linalg.norm(block_pos - goal_pos) <= 0.05:
            random_goal_x_pos = np.random.uniform(0.4, 0.6)
            random_goal_y_pos = np.random.uniform(-0.2, 0.2)
            random_goal_z_pos = np.random.uniform(0.45, 0.66)
            goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        site_id = self.get_site_id('goal_site')
        self.mj_model.site_pos[site_id] = goal_pos


if __name__ == "__main__":
    env = PickAndPlaceEnv()
    obs, _ = env.reset()
    for t in range(int(1e6)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
