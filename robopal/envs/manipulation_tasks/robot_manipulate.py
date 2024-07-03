import mujoco
import numpy as np
import logging
from typing import Dict, Union, Tuple, Any
from robopal.envs import RobotEnv
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)


class ManipulateEnv(RobotEnv):
    """
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=None,
                 render_mode='human',
                 control_freq=20,
                 controller='CARTIK',
                 is_interpolate=False,
                 action_type="velocity",
                 is_randomize_end=True,
                 is_randomize_object=True,
                 is_show_camera_in_cv=False,
                 is_render_camera_offscreen = False,
                 camera_in_render="frontview",
                 camera_in_window="free",
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            controller=controller,
            is_interpolate=is_interpolate,
            is_show_camera_in_cv=is_show_camera_in_cv,
            is_render_camera_offscreen=is_render_camera_offscreen,
            camera_in_render=camera_in_render,
            camera_in_window=camera_in_window,
        )

        self.action_type = action_type
        self.is_randomize_end = is_randomize_end
        self.is_randomize_object = is_randomize_object

        self.max_episode_steps = 50

        self.obs_dim: np.ndarray = None
        self.action_dim: np.ndarray = None

        self._timestep = 0
        self.goal_pos = None
        self.desired_position = self.init_pos[self.agents[0]]
        self.action_scale = 0.1

        self.grip_max_bound = self.robot.end[self.agents[0]]._ctrl_range[1]
        self.grip_min_bound = self.robot.end[self.agents[0]]._ctrl_range[0]

    def compute_end_position(self, input) -> Tuple[np.ndarray, Any]:
        """ Map to target action space bounds
        """
        if self.action_type == "velocity":
            actual_pos = self.desired_position + self.action_scale * input
        elif self.action_type == "position":
            actual_pos = input
        else:
            raise ValueError(f"Invalid action type: {self.action_type}")
        
        actual_pos = actual_pos.clip(self.robot.pos_min_bound, self.robot.pos_max_bound)
        return actual_pos
    
    def step(self, action) -> Tuple:
        """ Take one step in the environment.

        :param action:  The action space is 4-dimensional, with the first 3 dimensions corresponding to the desired
        linear velocities of the end effector in Cartesian coordinates, and the last dimension corresponding to the
        desired gripper state (0 denotes closed, 1 denotes open).
        :return: obs, reward, terminated, truncated, info
        """
        self._timestep += 1

        # normalized actions should be un-normalized before applying to the environment
        end_pos = self.compute_end_position(action[:3])
        
        # take one step
        normalized_gripper_ctrl = action[3]
        unnormalized_gripper_ctrl = (normalized_gripper_ctrl + 1) * (self.grip_max_bound - self.grip_min_bound) / 2 + self.grip_min_bound
        self.robot.end[self.agents[0]].apply_action(unnormalized_gripper_ctrl)
        super().step(end_pos)

        self.desired_position = end_pos

        obs = self._get_obs()
        reward = self.compute_rewards()
        terminated = False
        truncated = True if self._timestep >= self.max_episode_steps else False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_rewards(self, achieved_goal: np.ndarray = np.zeros(3), desired_goal: np.ndarray = np.zeros(3), info: dict = None, **kwargs):
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        if kwargs:
            return -(d >= kwargs['th']).astype(np.float64)
        return -(d >= 0.02).astype(np.float64)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, th=0.02) -> np.ndarray:
        """ Compute whether the achieved goal successfully achieved the desired goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < th).astype(np.float32)

    def _get_obs(self, agent: str = None) -> Union[Dict, np.ndarray]:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        raise NotImplementedError
    
    def _get_achieved_goal(self) -> np.ndarray:
        """ get achieved goal, required for goal-based env.
        """
        pass

    def _get_desired_goal(self) -> np.ndarray:
        """ get desired goal, required for goal-based env.
        """
        pass

    def _get_info(self, agent: str = None) -> dict:
        return {}

    def reset(self, seed=None, options=None):
        options = options or {}
        options['disable_reset_render'] = True
        
        super().reset(seed, options)

        self._timestep = 0

        if self.is_randomize_end:
            self.set_random_init_position()
        self.update_init_pose_to_current()

        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def update_init_pose_to_current(self):
        super().update_init_pose_to_current()

        # reset the desired position to the initial position
        self.desired_position = self.init_pos[self.agents[0]]

    def reset_object(self):
        """ Reset the object to a random pose within the workspace.
        """
        if self.is_randomize_object:
            pass
        return super().reset_object()

    def set_random_init_position(self):
        """ Set the initial position of the end effector to a random position within the workspace.
        """
        for agent in self.agents:
            random_pos = np.random.uniform(self.robot.pos_min_bound, self.robot.pos_max_bound)
            qpos = self.controller.ik(random_pos, np.array([1, 0, 0, 0]), q_init=self.robot.get_arm_qpos(agent))
            self.set_joint_qpos(qpos, agent)
            self.forward()
