import mujoco
import numpy as np
import logging
from typing import Dict, Union, Tuple, Any
from robopal.envs import RobotEnv
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)


class BimanualManipulate(RobotEnv):
    """
    The control frequency of the robot is of f = 20 Hz. This is achieved by applying the same action
    in 50 subsequent simulator step (with a time step of dt = 0.001 s) before returning the control to the robot.
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
        self.pos_max_bound = {self.agents[0]: np.array([0.65, 0.2, 0.4]),
                              self.agents[1]: np.array([0.65, 0.2, 0.4])}
        self.pos_min_bound = {self.agents[0]: np.array([0.3, -0.2, 0.14]),
                              self.agents[1]: np.array([0.3, -0.2, 0.14])}

    def action_normalize(self, action, agent) -> Tuple[np.ndarray, Any]:
        """
        Map to target action space bounds
        """
        current_pos, _ = self.controller.forward_kinematics(self.robot.get_arm_qpos(agent), agent=agent)
        next_pos = current_pos + self.pos_ratio * action[:3]
        next_pos = next_pos.clip(self.pos_min_bound[agent], self.pos_max_bound[agent])
        gripper_ctrl = (
            (action[3] + 1) 
            * (self.robot.end[agent]._ctrl_range[1] - self.robot.end[agent]._ctrl_range[0]) / 2 
            + self.robot.end[agent]._ctrl_range[0]
        )
        return next_pos, gripper_ctrl

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """ Take one step in the environment.

        :param action:  The action space is 4-dimensional, with the first 3 dimensions corresponding to the desired
        position of the block in Cartesian coordinates, and the last dimension corresponding to the
        desired gripper opening (0 for closed, 1 for open).
        :return: obs, reward, terminated, truncated, info
        """
        self._timestep += 1

        arms_actions = {agent: None for agent in self.agents}
        grippers_actions = {agent: None for agent in self.agents}
        for agent in self.agents:
            arms_actions[agent], grippers_actions[agent] = self.action_normalize(
                actions[agent], agent)
            
        # take one step
        for agent in self.agents:
            self.robot.end[agent].apply_action(grippers_actions[agent])
        super().step(arms_actions)

        observations = {agent: self._get_obs(agent) for agent in self.agents}

        rewards = {agent: self.compute_rewards(agent) for agent in self.agents}
        # Check termination conditions
        terminations = {agent: False for agent in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {agent: False for agent in self.agents}
        if self._timestep > self.max_episode_steps:
            truncations = {agent: True for agent in self.agents}

        infos = {agent: self._get_info(agent) for agent in self.agents}

        # if any(terminations.values()) or all(truncations.values()):
        #     self.agents = []

        return observations, rewards, terminations, truncations, infos

    def compute_rewards(self, agent: str):
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
    
    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _get_obs(self, agent: str = None) -> Union[Dict, np.ndarray]:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        raise NotImplementedError

    def _get_info(self, agent: str = None) -> dict:
        return {}

    def reset(self, seed=None, options=None):
        options = options or {}
        options['disable_reset_render'] = True
        super().reset(seed, options)

        self._timestep = 0
        # self.set_random_init_position()

        observations = {
            agent: self._get_obs(agent)
            for agent in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def reset_object(self):
        pass

    def set_random_init_position(self):
        """ Set the initial position of the end effector to a random position within the workspace.
        """
        for agent in self.agents:
            random_pos = np.random.uniform(self.pos_min_bound[agent], self.pos_max_bound[agent])
            qpos = self.controller.ik(random_pos, self.init_quat[agent], q_init=self.robot.get_arm_qpos(agent))
            self.set_joint_qpos(qpos, agent)
            self.forward()
            self.render()
