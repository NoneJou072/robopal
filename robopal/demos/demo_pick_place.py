import numpy as np
import logging

from robopal.envs.task_ik_ctrl_env import PosCtrlEnv
import robopal.commons.transform as trans

logging.basicConfig(level=logging.DEBUG)


class PickAndPlaceEnv(PosCtrlEnv):
    """ Reference: https://robotics.farama.org/envs/fetch/pick_and_place/#
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=200,
                 enable_camera_viewer=False,
                 cam_mode='rgb',
                 jnt_controller='IMPEDANCE',
                 is_interpolate=False,
                 is_pd=False,
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            cam_mode=cam_mode,
            jnt_controller=jnt_controller,
            is_interpolate=is_interpolate,
            is_pd=is_pd,
        )
        self.obs_dim = 16
        self.action_dim = 4
        self.max_action = 1.0
        self.min_action = -1.0
        self.max_episode_steps = 150

        self._timestep = 0

    def step(self, action):
        self._timestep += 1
        # Map to target action space bounds
        pos_ctrl = (action[3] + 1) * (0.0115 - (-0.015)) / 2 + (-0.015)

        pos_offset = 0.05 * action[:3]
        actual_pos_action = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)[0] + pos_offset

        pos_max_bound = np.array([0.6, 0.2, 0.4])
        pos_min_bound = np.array([0.3, -0.2, 0.14])
        actual_pos_action = actual_pos_action.clip(pos_min_bound, pos_max_bound)

        # take one step
        # self.gripper_ctrl('0_gripper_l_finger_joint', int(actual_action[3]))
        self.mj_data.joint('0_r_finger_joint').qpos[0] = pos_ctrl

        logging.debug(f'des_pos:{actual_pos_action[:3]}')
        super().step(actual_pos_action[:3])
        logging.debug(f'cur_pos:{self.kdl_solver.fk(self.robot.single_arm.arm_qpos)[0]}')

        obs = self.get_observations()
        reward = self.compute_rewards()
        terminated = False
        truncated = False
        if self._timestep >= self.max_episode_steps:
            truncated = True
        info = self.get_info()

        return obs, reward, terminated, truncated, info

    def inner_step(self, action):
        super().inner_step(action)

    def compute_rewards(self) -> float | int:
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        reward = -1
        obs = self.get_observations()
        obj_pos = obs[:3]
        goal_pos = obs[3:6]
        dist = np.linalg.norm(obj_pos - goal_pos)
        if dist < 0.05:
            reward = 0
        return reward

    def get_observations(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim)
        obs[:3] = self.get_body_pos('green_block')
        obs[3:6] = self.get_body_pos('goal_site')
        obs[6:9] = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)[0]
        obs[9:12] = obs[6:9] - obs[:3]
        obs[12] = self.mj_data.actuator("0_gripper_l_finger_joint").ctrl[0]
        obs[13:16] = trans.mat_2_euler(self.get_body_rotm('green_block'))
        return obs

    def get_info(self) -> dict:
        return {}

    def reset(self):
        self._timestep = 0
        super().reset()


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaGrasp

    env = PickAndPlaceEnv(
        robot=DianaGrasp(),
        renderer="viewer",
        is_render=True,
        control_freq=10,
        is_interpolate=False,
        is_pd=False,
        jnt_controller='IMPEDANCE',
    )
    env.reset()

    for t in range(int(1e6)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, _ = env.step(action)
        if truncated:
            env.reset()
        env.render()
