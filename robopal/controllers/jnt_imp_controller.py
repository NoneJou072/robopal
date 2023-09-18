import numpy as np
from robopal.commons.pin_utils import PinSolver


class Jnt_Impedance(object):
    def __init__(self, robot):
        self.kdl_solver = PinSolver(robot.urdf_path)

        # hyper-parameters of impedance controller
        self.Bj = np.zeros(7)
        self.kj = np.zeros(7)

        self.set_jnt_params(
            b=47.0 * np.ones(7),
            k=200.0 * np.ones(7),
        )
        print("Jnt_Impedance Initialized!")

    def set_jnt_params(self, b: np.ndarray, k: np.ndarray):
        self.Bj = b
        self.kj = k

    def compute_jnt_torque(
            self,
            q_des: np.ndarray,
            v_des: np.ndarray,
            q_cur: np.ndarray,
            v_cur: np.array,
    ):
        """ robot的关节空间控制的计算公式
            Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_des: desired joint position
        :param v_des: desired joint velocity
        :param q_cur: current joint position
        :param v_cur: current joint velocity
        :return: desired joint torque
        """
        M = self.kdl_solver.get_inertia_mat(q_cur)
        C = self.kdl_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kdl_solver.get_gravity_mat(q_cur)
        coriolis_gravity = C[-1] + g

        acc_desire = self.kj * (q_des - q_cur) + self.Bj * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau
