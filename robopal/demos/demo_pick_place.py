import numpy as np
import logging

from robopal.envs.task_ik_ctrl_env import PosCtrlEnv
import robopal.commons.transform as trans

logging.basicConfig(level=logging.INFO)


class PickAndPlaceEnv(PosCtrlEnv):
    """ Reference: https://robotics.farama.org/envs/fetch/pick_and_place/#
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 render_mode='human',
                 control_freq=200,
                 enable_camera_viewer=False,
                 cam_mode='rgb',
                 jnt_controller='JNTIMP',
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
        self.name = 'PickAndPlace-v1'

        self.obs_dim = (22,)
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50
        self._timestep = 0

        self.goal_pos = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(self, action) -> tuple:
        """ Take one step in the environment.

        :param action:  The action space is 4-dimensional, with the first 3 dimensions corresponding to the desired
        position of the block in Cartesian coordinates, and the last dimension corresponding to the
        desired gripper opening (0 for closed, 1 for open).
        :return: obs, reward, terminated, truncated, info
        """
        self._timestep += 1

        pos_offset = 0.05 * action[:3]
        actual_pos_action = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)[0] + pos_offset

        pos_max_bound = np.array([0.6, 0.2, 0.4])
        pos_min_bound = np.array([0.3, -0.2, 0.12])
        actual_pos_action = actual_pos_action.clip(pos_min_bound, pos_max_bound)

        # Map to target action space bounds
        gripper_ctrl = (action[3] + 1) * (0.0115 - (-0.01)) / 2 + (-0.01)
        # take one step
        self.mj_data.joint('0_r_finger_joint').qpos[0] = gripper_ctrl
        self.mj_data.joint('0_l_finger_joint').qpos[0] = gripper_ctrl

        logging.debug(f'des_pos:{actual_pos_action[:3]}')
        super().step(actual_pos_action[:3])
        logging.debug(f'cur_pos:{self.kdl_solver.fk(self.robot.single_arm.arm_qpos)[0]}')

        obs = self._get_obs()
        reward = self.compute_rewards(obs['achieved_goal'], obs['desired_goal'])
        terminated = False
        truncated = True if self._timestep >= self.max_episode_steps else False
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def inner_step(self, action):
        super().inner_step(action)

    def compute_rewards(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict = None):
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        assert achieved_goal.shape == desired_goal.shape
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist > 0.05).astype(np.float32)

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[:3] = (  # position of the block
            object_pos := self.get_body_pos('green_block')
        )
        obs[3:6] = (  # position of the end
            end_pos := self.kdl_solver.fk(self.robot.single_arm.arm_qpos)[0]
        )
        obs[6:9] = end_pos - object_pos  # distance between the block and the end
        obs[9:12] = trans.mat_2_euler(self.get_body_rotm('green_block'))
        obs[12:15] = (  # End effector linear velocity
            end_vel := self.kdl_solver.get_end_vel(self.robot.single_arm.arm_qpos, self.robot.single_arm.arm_qvel)[:3]
        )
        # velocity with respect to the gripper
        dt = 0.0005
        object_velp = self.get_body_xvelp('green_block')
        object2end_velp = object_velp - end_vel
        obs[15:18] = object2end_velp

        obs[18:21] = self.get_body_xvelr('green_block')
        obs[21] = self.mj_data.joint('0_r_finger_joint').qpos[0]

        return {
            'observation': obs.copy(),
            'achieved_goal': object_pos.copy(),  # the current state of the block
            'desired_goal': self.goal_pos.copy()
        }

    def _get_info(self) -> dict:
        return {}

    def reset(self, seed=None):
        super().reset()
        self._timestep = 0
        # set new goal
        self.goal_pos = self.get_body_pos('goal_site')

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, info


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaGrasp

    env = PickAndPlaceEnv(
        robot=DianaGrasp(),
        renderer="viewer",
        is_render=True,
        control_freq=10,
        is_interpolate=False,
        is_pd=False,
        jnt_controller='JNTIMP',
    )
    env.reset()

    for t in range(int(1e6)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, _ = env.step(action)
        if truncated:
            env.reset()
