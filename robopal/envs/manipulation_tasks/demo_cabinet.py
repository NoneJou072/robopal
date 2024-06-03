import numpy as np

from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
import robopal.commons.transform as trans
from robopal.robots.diana_med import DianaCabinet
from robopal.wrappers import GoalEnvWrapper


class LockedCabinetEnv(ManipulateEnv):

    name = 'LockedCabinet-v1'
    
    def __init__(self,
                 robot=DianaCabinet,
                 render_mode='human',
                 control_freq=20,
                 is_show_camera_in_cv=False,
                 controller='CARTIK',
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            is_show_camera_in_cv=is_show_camera_in_cv,
            controller=controller,
        )

        self.obs_dim = (38,)
        self.goal_dim = (9,)
        self.action_dim = (4,)

        self.max_action = 1.0
        self.min_action = -1.0

        self.max_episode_steps = 50

        self.TASK_FLAG = 0

        self.pos_max_bound = np.array([0.6, 0.25, 0.4])
        self.pos_min_bound = np.array([0.4, -0.25, 0.2])

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

        return obs.copy()

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
                self.get_site_pos('beam_left'), self.get_site_pos('cabinet_mid'), th=0.03
            ) == 0 else self.get_site_pos('left_handle'),
            self.get_site_pos('cabinet_mid'),
            self.get_site_pos('cabinet_left_opened'),
        ], axis=0)
        return desired_goal.copy()

    def _get_info(self) -> dict:
        return {
            'is_unlock_success': self._is_success(self.get_site_pos('beam_left'), self.get_site_pos('cabinet_mid'), th=0.03),
            'is_door_success': self._is_success(self.get_site_pos('left_handle'), self.get_site_pos('cabinet_left_opened'), th=0.03)
        }

    def reset_object(self):
        if self.TASK_FLAG == 0:
            pass
        elif self.TASK_FLAG == 1:
            self.mj_data.joint('OBJTy').qpos[0] = -0.12

        return super().reset_object()


if __name__ == "__main__":
    env = LockedCabinetEnv()
    env = GoalEnvWrapper(env)
    env.reset()
    for t in range(int(1e5)):
        action = np.random.uniform(env.min_action, env.max_action, env.action_dim)
        s_, r, terminated, truncated, info = env.step(action)
        if truncated:
            env.reset()
    env.close()
