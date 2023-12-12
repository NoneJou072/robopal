import numpy as np
from robopal.envs.robot import RobotEnv
import robopal.commons.transform as T


class PosCtrlEnv(RobotEnv):
    def __init__(self,
                 robot=None,
                 control_freq=200,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 is_pd=False,
                 render_mode='human',
                 ):
        super().__init__(
            robot=robot,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
            is_interpolate=is_interpolate,
            render_mode=render_mode,
        )
        self.p_cart = 0.2
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01
        self.is_pd = is_pd
        self.vel_des = np.zeros(3)

        _, self.init_rot_quat = self.kdl_solver.fk(self.robot.arm_qpos, rot_format='quaternion')

    def compute_pd_increment(self, p_goal: np.ndarray,
                             p_cur: np.ndarray,
                             r_goal: np.ndarray,
                             r_cur: np.ndarray,
                             pd_goal: np.ndarray = np.zeros(3),
                             pd_cur: np.ndarray = np.zeros(3)):
        pos_incre = self.p_cart * (p_goal - p_cur) + self.d_cart * (pd_goal - pd_cur)
        quat_incre = self.p_quat * (r_goal - r_cur)
        return pos_incre, quat_incre

    def step_controller(self, action):
        if len(action) not in (3, 7):
            raise ValueError("Invalid action length.")
        if not self.is_pd:
            p_goal = action[:3]
            r_goal = T.quat_2_mat(self.init_rot_quat if len(action) == 3 else action[3:])
        else:
            p_cur, r_cur = self.kdl_solver.fk(self.robot.arm_qpos, rot_format='quaternion')

            r_target = self.init_rot_quat if len(action) == 3 else action[3:]
            pd_cur = self.kdl_solver.get_end_vel(self.robot.arm_qpos, self.robot.arm_qvel)
            p_incre, r_incre = self.compute_pd_increment(p_goal=action[:3], p_cur=p_cur,
                                                         r_goal=r_target, r_cur=r_cur,
                                                         pd_goal=self.vel_des, pd_cur=pd_cur[:3])
            p_goal = p_incre + p_cur
            r_goal = T.quat_2_mat(r_cur + r_incre)

        return self.kdl_solver.ik(p_goal, r_goal, q_init=self.robot.arm_qpos)

    def step(self, action):
        action = self.step_controller(action)
        super().step(action)


if __name__ == "__main__":
    from robopal.robots.diana_med import DianaMed

    env = PosCtrlEnv(
        robot=DianaMed(),
        render_mode='human',
        control_freq=20,
        is_interpolate=False,
        is_pd=False,
        controller='JNTIMP'
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.33116, -0.09768533, 0.26947228, 1, 0, 0, 0])
        env.step(action)
        if env.is_render:
            env.render()
