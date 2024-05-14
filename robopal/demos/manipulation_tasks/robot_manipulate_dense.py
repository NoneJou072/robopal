import mujoco
import numpy as np
import logging

from robopal.envs import RobotEnv
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)


class ManipulateDenseEnv(RobotEnv):
    """
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=None,
                 render_mode='human',
                 control_freq=20,
                 enable_camera_viewer=False,
                 controller='CARTIK',
                 is_interpolate=False,
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
            is_interpolate=is_interpolate,
        )

        self.max_episode_steps = 50

        self._timestep = 0
        self.goal_pos = None

        self.pos_ratio = 0.1
        self.pos_max_bound = np.array([0.65, 0.2, 0.4])
        self.pos_min_bound = np.array([0.3, -0.2, 0.14])
        self.grip_max_bound = 0.02
        self.grip_min_bound = -0.02

    def action_scale(self, action):
        """
        Map to target action space bounds
        """
        current_pos, _ = self.controller.forward_kinematics(self.robot.get_arm_qpos())
        actual_pos_action = current_pos + self.pos_ratio * action[:3]
        actual_pos_action = actual_pos_action.clip(self.pos_min_bound, self.pos_max_bound)
        gripper_ctrl = (action[3] + 1) * (self.grip_max_bound - self.grip_min_bound) / 2 + self.grip_min_bound
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
        self.mj_data.actuator('0_robotiq_2f_85').ctrl[0] = gripper_ctrl

        super().step(actual_pos_action[:3])

        obs = self._get_obs()
        reward = self.compute_rewards()
        terminated = False
        truncated = True if self._timestep >= self.max_episode_steps else False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def compute_rewards(self, info: dict = None, **kwargs):
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        return 0

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, th=0.02) -> np.ndarray:
        """ Compute whether the achieved goal successfully achieved the desired goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < th).astype(np.float32)
    
    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        return {}

    def _get_info(self) -> dict:
        return {}

    def reset(self, seed=None, options=None):
        options = options or {}
        options['disable_reset_render'] = True
        super().reset(seed, options)
        # self.set_random_init_position()
        self._timestep = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def reset_object(self):
        pass

    def set_random_init_position(self):
        """ Set the initial position of the end effector to a random position within the workspace.
        """
        for agent in self.robot.agents:
            random_pos = np.random.uniform(self.pos_min_bound, self.pos_max_bound)
            qpos = self.controller.ik(random_pos, self.init_quat[agent], q_init=self.robot.get_arm_qpos(agent))
            self.set_joint_qpos(qpos, agent)
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.render()
