import numpy as np

from robopal.commons.pin_utils import PinSolver
import robopal.commons.transform as trans


class JntIK:
    """
    ik
    """

    def __init__(
            self,
            robot,
            is_pd=False,
            **kwargs
    ):
        self.name = 'CARTIMP'
        self.dofs = 7
        self.robot = robot

        self.p_cart = 0.2
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01

        self.is_pd = is_pd

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
            p_cur, r_cur = self.kd_solver.fk(self.robot.get_arm_qpos(), rot_format='quaternion')

            r_target = self.init_rot_quat if len(action) == 3 else action[3:]
            pd_cur = self.kd_solver.get_end_vel(self.robot.get_arm_qpos(), self.robot.get_arm_qvel())
            p_incre, r_incre = self.compute_pd_increment(p_goal=action[:3], p_cur=p_cur,
                                                         r_goal=r_target, r_cur=r_cur,
                                                         pd_goal=self.vel_des, pd_cur=pd_cur[:3])
            p_goal = p_incre + p_cur
            r_goal = T.quat_2_mat(r_cur + r_incre)

        return self.kd_solver.ik(p_goal, r_goal, q_init=self.robot.get_arm_qpos())
