import numpy as np
import logging

from robopal.envs.task_ik_ctrl_env import PosCtrlEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaCabinet

logging.basicConfig(level=logging.INFO)


class LockedCabinetEnv(PosCtrlEnv):
    """
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=DianaCabinet(),
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
        self.name = 'LockedCabinet-v1'

        self.obs_dim = (38,)
        self.goal_dim = (9,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50
        self._timestep = 0

        self.TASK_FLAG = 0

    def action_scale(self, action):
        pos_offset = 0.1 * action[:3]
        actual_pos_action = self.kdl_solver.fk(self.robot.arm_qpos)[0] + pos_offset

        pos_max_bound = np.array([0.6, 0.25, 0.4])
        pos_min_bound = np.array([0.4, -0.25, 0.2])
        actual_pos_action = actual_pos_action.clip(pos_min_bound, pos_max_bound)

        # Map to target action space bounds
        grip_max_bound = 0.02
        grip_min_bound = -0.02
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
        achieved_goal = obs['achieved_goal']
        desired_goal = obs['desired_goal']
        reward = self.compute_rewards(achieved_goal, desired_goal, th=0.03)
        terminated = False
        truncated = True if self._timestep >= self.max_episode_steps else False
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

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
        obs[3:6] = (  # handle position in global coordinates
            handle_pos := self.get_site_pos('left_handle')
        )
        obs[6:9] = (  # beam position in global coordinates
            beam_pos := self.get_site_pos('beam_left')
        )
        obs[9:12] = end_pos - handle_pos  # distance between the handle and the end
        obs[12:15] = end_pos - beam_pos  # distance between the beam and the end
        obs[15:18] = trans.mat_2_euler(self.get_site_rotm('left_handle'))
        obs[18:21] = trans.mat_2_euler(self.get_site_rotm('beam_left'))
        obs[21:24] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * self.dt
        )
        # velocity with respect to the gripper
        handle_velp = self.get_site_xvelp('left_handle') * self.dt
        obs[24:27] = (  # velocity with respect to the gripper
            handle_velp - end_vel
        )
        beam_velp = self.get_site_xvelp('beam_left') * self.dt
        obs[27:30] = (  # velocity with respect to the gripper
            beam_velp - end_vel
        )
        obs[30:33] = self.get_site_xvelr('left_handle') * self.dt
        obs[33:36] = self.get_site_xvelr('beam_left') * self.dt
        obs[36] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[37] = self.mj_data.joint('0_r_finger_joint').qvel[0] * self.dt

        return {
            'observation': obs.copy(),
            'achieved_goal': self._get_achieved_goal(),
            'desired_goal': self._get_desired_goal()
        }

    def _get_achieved_goal(self):
        achieved_goal = np.concatenate([
            self.get_site_pos('0_grip_site'),
            self.get_site_pos('beam_left'),
            self.get_site_pos('left_handle')
        ], axis=0)
        return achieved_goal.copy()

    def _get_desired_goal(self):
        desired_goal = np.concatenate([
            self.get_site_pos('beam_right') if self._is_success(
                self.get_site_pos('beam_left'), self.get_site_pos('cabinet_mid'), th=0.05
            ) == 0 else self.get_site_pos('left_handle'),
            self.get_site_pos('cabinet_mid'),
            self.get_site_pos('cabinet_left_opened'),
        ], axis=0)
        return desired_goal.copy()

    def _get_info(self) -> dict:
        return {
            'is_unlock_success': self._is_success(self.get_site_pos('beam_left'), self.get_site_pos('cabinet_mid'), th=0.05),
            'is_door_success': self._is_success(self.get_site_pos('left_handle'), self.get_site_pos('cabinet_left_opened'), th=0.03)
        }

    def reset(self, seed=None):
        super().reset()
        self._timestep = 0

        if self.TASK_FLAG == 0:
            pass
        elif self.TASK_FLAG == 1:
            self.mj_data.joint('OBJTy').qpos[0] = -0.12

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, info

    def compute_rewards(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, **kwargs) -> np.ndarray:
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        assert 'th' in kwargs.keys()
        d = self.goal_distance(achieved_goal, desired_goal)
        return -(d > kwargs['th']).astype(np.float64)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, th=0.03) -> np.ndarray:
        """ Compute whether the achieved goal successfully achieved the desired goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < th).astype(np.float32)

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)


if __name__ == "__main__":

    env = LockedCabinetEnv()
    # env.TASK_FLAG = 1
    env.reset()
    for t in range(int(1e6)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        # beam_right_pos = env.get_site_pos('beam_right') - np.array([0, 0, 0.32])
        # action = np.concatenate([beam_right_pos,np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
        print(env.goal_distance(env.get_site_pos('beam_left'), env.get_site_pos('cabinet_mid')))
        # if truncated:
        #     env.reset()
    a = None
    for t in range(int(1e2)):
        drawer_pos = env.get_site_pos('beam_right') - np.array([0, 0, 0.32])
        action = np.concatenate([drawer_pos,-1.0 * np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
        a = drawer_pos.copy()
    a[1] -= 0.11
    for t in range(int(1e2)):
        action = np.concatenate([a,-1.0 * np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
    for t in range(int(1e2)):
        action = np.concatenate([a,1.0 * np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
    for t in range(int(1e2)):
        beam_right_pos = env.get_site_pos('left_handle') - np.array([0, 0, 0.2])
        action = np.concatenate([beam_right_pos,np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
    for t in range(int(1e2)):
        beam_right_pos = env.get_site_pos('left_handle') - np.array([0, 0, 0.32])
        action = np.concatenate([beam_right_pos,np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
    for t in range(int(1e2)):
        beam_right_pos = env.get_site_pos('left_handle') - np.array([0, 0, 0.32])
        action = np.concatenate([beam_right_pos,-1.0 * np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
    for t in range(int(1e6)):
        beam_right_pos = env.get_site_pos('cabinet_left_opened') - np.array([0, 0, 0.32])
        action = np.concatenate([beam_right_pos, -1.0 * np.ones(1)])
        s_, r, terminated, truncated, _ = env.step(action)
