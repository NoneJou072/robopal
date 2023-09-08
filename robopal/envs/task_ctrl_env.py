import numpy as np
from robopal.envs.jnt_ctrl_env import SingleArmEnv
import robopal.commons.transform as T


class PosCtrlEnv(SingleArmEnv):
    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=200,
                 is_interpolate=True,
                 is_pd=False,
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate
        )
        self.p_cart = 0.2
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01
        self.is_pd = is_pd
        self._vel_des = np.zeros(3)

        _, r_init_m = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)
        self.init_rot_quat = T.mat_2_quat(r_init_m)

    @property
    def vel_cur(self):
        """ Current velocity, consist of 3*1 cartesian and 4*1 quaternion """
        j = self.kdl_solver.get_jac(self.robot.single_arm.arm_qpos)
        vel_cur = np.dot(j, self.robot.single_arm.arm_qvel)
        return vel_cur

    @property
    def vel_des(self):
        """ Desired velocity of 3*1 cartesian position"""
        return self._vel_des

    @vel_des.setter
    def vel_des(self, value):
        self._vel_des = value

    def PDControl(self, p_goal, p_cur, r_goal, r_cur, vel_goal=np.zeros(3)):
        pos_incre = self.p_cart * (p_goal - p_cur) + self.d_cart * (vel_goal - self.vel_cur[:3])
        quat_incre = self.p_quat * (r_goal - r_cur)
        return pos_incre, quat_incre

    def step(self, action):
        if len(action) != 3 and len(action) != 7:
            raise ValueError("Fault action length.")
        if self.is_pd is False:
            p_goal = action[:3]
            r_goal = T.quat_2_mat(self.init_rot_quat if len(action) == 3 else action[3:])
        else:
            p_cur, r_cur_m = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)
            r_cur = T.mat_2_quat(r_cur_m)
            r_target = self.init_rot_quat if len(action) == 3 else action[3:]
            p_incre, r_incre = self.PDControl(p_goal=action[:3], p_cur=p_cur,
                                        r_goal=r_target, r_cur=r_cur, vel_goal=self.vel_des)
            p_goal = p_incre + p_cur
            r_goal = T.quat_2_mat(r_cur + r_incre)

        action = self.kdl_solver.ik(p_goal, r_goal, q_init=self.robot.single_arm.arm_qpos)
        return super().step(action)


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = PosCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=200,
        is_interpolate=False,
        is_pd=True
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.33116, -0.09768533, 0.26947228, 1, 0, 0, 0])
        env.step(action)
        if env.is_render:
            env.render()
