import numpy as np
from robopal.envs.task_ik_ctrl_env import PosCtrlEnv
import robopal.commons.transform as trans

TRAIN_MODE = False


class PickAndPlaceEnv(PosCtrlEnv):
    """ Reference: https://robotics.farama.org/envs/fetch/pick_and_place/#
    The control frequency of the robot is of f = 25 Hz. This is achieved by applying the same action
    in 40 subsequent simulator step (with a time step of dt = 0.0005 s) before returning the control to the robot.
    """

    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=20,
                 is_interpolate=True,
                 is_pd=True,
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate,
            is_pd=is_pd
        )
        self.obs_dim = 16
        self.action_dim = 4
        self.max_action = 1.0
        self.min_action = -1.0
        self.max_episode_steps = 50

        self._timestep = 0

    def step(self, action):
        self._timestep += 1
        # Map to target action space bounds
        actual_min_action = np.array([0.1, -0.5, 0.15, 0.0])
        actual_max_action = np.array([0.6, 0.5, 0.5, 1.0])
        mapped_action = (action + 1) * (actual_max_action - actual_min_action) / 2 + actual_min_action

        # take one step
        self.gripper_ctrl('0_gripper_l_finger_joint', int(mapped_action[3]))
        super().step(mapped_action[:3])

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
        control_freq=25,
        is_interpolate=False,
        is_pd=False
    )
    env.reset()

    if TRAIN_MODE is False:
        for t in range(int(1e6)):
            action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
            s_, r, terminated, truncated, _ = env.step(action)
            if env.is_render:
                env.render()
            if truncated:
                env.reset()

    else:
        from robopal.commons.gym_wrapper import GymWrapper
        env = GymWrapper(env)
        del env
