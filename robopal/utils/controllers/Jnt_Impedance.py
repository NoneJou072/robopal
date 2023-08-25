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
        # robot的关节空间控制的计算公式（multiply等同于向量相乘）
        tau = np.multiply(self.kj, q_des - q_cur) - np.multiply(self.Bj, v_cur)
        tau = np.dot(kwargs['M'], tau)  # 乘上质量矩阵会更稳定一些
        tau += kwargs['coriolis_gravity'][:7]  # 加上科氏力和重力矩
        return tau
