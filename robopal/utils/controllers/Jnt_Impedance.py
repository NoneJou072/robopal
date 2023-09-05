import numpy as np


class Jnt_Impedance(object):
    def __init__(self, robot):
        # hyper-parameters of impedance
        self.Mj = np.zeros(7)
        self.Bj = np.zeros(7)
        self.kj = np.zeros(7)

        self.set_jnt_params(
            m=0.1 * np.diag(np.ones(6)),
            b=30.0 * np.ones(7),
            k=100.0 * np.ones(7),
        )
        print("Jnt_Impedance Initialized!")

    def set_jnt_params(self, m: np.ndarray, b: np.ndarray, k: np.ndarray):
        self.Mj = m
        self.Bj = b
        self.kj = k

    def compute_jnt_torque(
            self,
            q_des: np.ndarray,
            v_des: np.ndarray,
            q_cur: np.ndarray,
            v_cur: np.array,
            *args,
            **kwargs
    ):
        """ robot的关节空间控制的计算公式
            Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_des: desired joint position
        :param v_des:
        :param q_cur:
        :param v_cur:
        :param args:
        :param kwargs:
        :return:
        """
        acc_desire = self.kj * (q_des - q_cur) + self.Bj * (v_des - v_cur)
        tau = np.dot(kwargs['M'], acc_desire) + kwargs['coriolis_gravity']
        return tau
