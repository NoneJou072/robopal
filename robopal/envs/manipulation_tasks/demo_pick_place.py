import numpy as np

from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaPickAndPlace
from robopal.wrappers import GoalEnvWrapper

class PickAndPlaceEnv(ManipulateEnv):

    name = 'PickAndPlace-v1'
    
    def __init__(self,
                 robot="PandaPickAndPlace",
                 render_mode='human',
                 control_freq=20,
                 is_show_camera_in_cv=False,
                 controller='CARTIK',
                 action_type="velocity",
                 is_render_camera_offscreen = False,
                 camera_in_render="frontview",
                 camera_in_window="free",
                 is_randomize_end=False,
                 is_randomize_object=True,
                 is_randomize_goal=False,
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            is_show_camera_in_cv=is_show_camera_in_cv,
            controller=controller,
            action_type=action_type,
            is_randomize_end=is_randomize_end,
            is_randomize_object=is_randomize_object,
            is_render_camera_offscreen=is_render_camera_offscreen,
            camera_in_render=camera_in_render,
            camera_in_window=camera_in_window,
        )

        self.obs_dim = (22,)
        self.goal_dim = (3,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.is_randomize_goal = is_randomize_goal

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)

        obs[0:15] = np.concatenate([  
            self.robot.get_arm_qpos(),
            # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site'),
            # gripper linear velocity
            self.get_site_xvelp('0_grip_site') * self.dt,
            self.robot.end['agent0'].get_finger_observations()
        ])
        obs[15:18] = (  # block position in global coordinates
            object_pos := self.get_body_pos('green_block')
        )
        obs[18:22] = (  # block rotation
            self.get_body_quat('green_block')
        )
        # obs[11:14] = (  # Relative block position with respect to gripper position in globla coordinates.
        #     end_pos - object_pos
        # )

        return obs.copy()
    
    def _get_achieved_goal(self) -> np.ndarray:
        return self.get_body_pos('green_block')

    def _get_desired_goal(self) -> np.ndarray:
        return self.get_site_pos('goal_site')
    
    def compute_rewards(self, achieved_goal: np.ndarray = np.zeros(3), desired_goal: np.ndarray = np.zeros(3), info: dict = None, **kwargs):
        """ Sparse Reward: the returned reward can have two values: -1 if the block hasn’t reached its final
        target position, and 0 if the block is in the final target position (the block is considered to have
        reached the goal if the Euclidean distance between both is lower than 0.05 m).
        """
        d = self.goal_distance(self._get_achieved_goal(), self._get_desired_goal())
        if kwargs:
            return -(d >= kwargs['th']).astype(np.float64)
        return -(d >= 0.02).astype(np.float64)
    
    def _get_info(self) -> dict:
        return {'is_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('goal_site'), th=0.02)}

    def reset_object(self):
        if self.is_randomize_object:
            random_x_pos, random_y_pos = np.random.uniform([0.35, -0.15], [0.55, 0.15])
            block_pose = np.array([random_x_pos, random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0])
            self.set_object_pose('green_block:joint', block_pose)
        else:
            self.set_object_pose('green_block:joint', np.array([0.5, 0.1, 0.46, 1.0, 0.0, 0.0, 0.0]))

        if self.is_randomize_goal:
            goal_pos = np.random.uniform([0.35, -0.15, 0.46], [0.55, 0.15, 0.65])
            while np.linalg.norm(block_pose[:3] - goal_pos) <= 0.05:
                goal_pos = np.random.uniform([0.35, -0.15, 0.46], [0.55, 0.15, 0.65])
            self.set_site_pos('goal_site', goal_pos)

        return super().reset_object()


if __name__ == "__main__":
    env = PickAndPlaceEnv(
        is_render_camera_offscreen=True,
        is_randomize_end=False,
        is_randomize_object=True,
    )
    env = GoalEnvWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
