import numpy as np

from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaTripleStack
from robopal.wrappers import GoalEnvWrapper


class TripleStackEnv(ManipulateEnv):

    name = 'MultiCubeStack-v1'
    
    def __init__(self,
                 robot=DianaTripleStack,
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

        self.obs_dim = (35,)
        self.goal_dim = (12,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.is_randomize_goal = is_randomize_goal

    def _get_obs(self):
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        The actual observation is format the table below.
        | gripper position | blocks position & rotation | gripper vel | gripper_qpos | gripper_qvel |
        """
        obs = np.zeros(self.obs_dim)

        # gripper state
        obs[0:8] = np.concatenate((
            self.get_site_pos('0_grip_site'),  # gripper position in global coordinates
            self.get_site_xvelp('0_grip_site') * self.dt,  # gripper linear velocity
            self.robot.end['agent0'].get_finger_observations()
        ))

        # red block state
        obs[8:17] = np.concatenate((
            self.get_body_pos('red_block'),  # block position
            trans.mat_2_euler(self.get_body_rotm('red_block')),  # block rotation
            self.get_body_xvelp('red_block') * self.dt,  # velocity
        ))

        # green block state
        obs[17:26] = np.concatenate((
            self.get_body_pos('green_block'),  # block position
            trans.mat_2_euler(self.get_body_rotm('green_block')),  # block rotation
            self.get_body_xvelp('green_block') * self.dt,  # velocity
        ))

        # blue block state
        obs[26:35] = np.concatenate((
            self.get_body_pos('blue_block'),  # block position
            trans.mat_2_euler(self.get_body_rotm('blue_block')),  # block rotation
            self.get_body_xvelp('blue_block') * self.dt,  # velocity
        ))

        return obs.copy()

    def _get_achieved_goal(self):
        achieved_goal = np.concatenate([
            self.get_body_pos('red_block'),
            self.get_body_pos('green_block'),
            self.get_body_pos('blue_block'),
        ], axis=0)
        return achieved_goal.copy()

    def _get_desired_goal(self):
        return np.concatenate([
            self.get_site_pos('red_goal'),
            self.get_site_pos('green_goal'),
            self.get_site_pos('blue_goal'),
        ], axis=0).copy()

    def _get_info(self):
        return {
            'is_success': self._is_success(self.get_body_pos('red_block'), self.get_site_pos('red_goal'), th=0.02)\
                and self._is_success(self.get_body_pos('green_block'), self.get_site_pos('green_goal'), th=0.02)\
                and self._is_success(self.get_body_pos('blue_block'), self.get_site_pos('blue_goal'), th=0.02)
        }

    def reset(self, seed=None, options=None):
        return super().reset(seed, options)

    def reset_object(self):
        # set the position of the red, green, and blue blocks
        if self.is_randomize_object:
            r_random_x_pos = np.random.uniform(0.3, 0.4)
            r_random_y_pos = np.random.uniform(-0.15, 0.15)
            self.set_object_pose('red_block:joint', np.array([r_random_x_pos, r_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

            g_random_x_pos = np.random.uniform(0.3, 0.4)
            g_random_y_pos = np.random.uniform(-0.15, 0.15)
            while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([g_random_x_pos, g_random_y_pos])) < 0.08:
                g_random_x_pos = np.random.uniform(0.3, 0.4)
                g_random_y_pos = np.random.uniform(-0.15, 0.15)
            self.set_object_pose('green_block:joint', np.array([g_random_x_pos, g_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

            b_random_x_pos = np.random.uniform(0.3, 0.4)
            b_random_y_pos = np.random.uniform(-0.15, 0.15)
            while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([b_random_x_pos, b_random_y_pos])) < 0.08 \
                or np.linalg.norm(np.array([g_random_x_pos, g_random_y_pos]) - np.array([b_random_x_pos, b_random_y_pos])) < 0.08:
                b_random_x_pos = np.random.uniform(0.3, 0.4)
                b_random_y_pos = np.random.uniform(-0.15, 0.15)
            self.set_object_pose('blue_block:joint', np.array([b_random_x_pos, b_random_y_pos, 0.46, 1.0, 0.0, 0.0, 0.0]))

        if self.is_randomize_goal:
            # red goal
            random_goal_x_pos = np.random.uniform(0.3, 0.4)
            random_goal_y_pos = np.random.uniform(-0.15, 0.15)
            while np.linalg.norm(np.array([r_random_x_pos, r_random_y_pos]) - np.array([random_goal_x_pos, random_goal_y_pos])) < 0.1 \
                or np.linalg.norm(np.array([g_random_x_pos, g_random_y_pos]) - np.array([random_goal_x_pos, random_goal_y_pos])) < 0.1 \
                or np.linalg.norm(np.array([b_random_x_pos, b_random_y_pos]) - np.array([random_goal_x_pos, random_goal_y_pos])) < 0.1:
                random_goal_x_pos = np.random.uniform(0.3, 0.4)
                random_goal_y_pos = np.random.uniform(-0.15, 0.15)
            red_goal = np.array([random_goal_x_pos, random_goal_y_pos, 0.44])
        else:
            red_goal = np.array([0.55, 0.1, 0.44])
        self.set_site_pos('red_goal', red_goal)

        # green goal
        green_goal = np.array([red_goal[0], red_goal[1], red_goal[2] + 0.04])
        self.set_site_pos('green_goal', green_goal)

        # blue goal
        blue_goal = np.array([green_goal[0], green_goal[1], green_goal[2] + 0.04])
        self.set_site_pos('blue_goal', blue_goal)
            
        return super().reset_object()

if __name__ == "__main__":
    env = TripleStackEnv()
    env = GoalEnvWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
