import numpy as np
from os.path import join, dirname
import os
from robopal.utils.KDL_utils import KDL_utils

work_space_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Jnt_Impedance(object):
    def __init__(self):
        urdf_path = join(dirname(__file__), "../../assets/models/manipulators/DianaMed/DianaMed.urdf")
        self.kdl_solver = KDL_utils(urdf_path)

        # hyper-parameters of impedance
        self.Mj = np.zeros(7)
        self.Bj = np.zeros(7)
        self.kj = np.zeros(7)

        self.init_qpos = [0, -1.6, 1.6, -1.6, -1.6, 0, 0]
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']

        print("Jnt_Impedance Initialized!")

    def set_jnt_Params(self, m: np.ndarray, b: np.ndarray, k: np.ndarray):
        """

        Args:
            m:
            b:
            k:

        Returns:

        """
        self.Mj = m
        self.Bj = b
        self.kj = k

    def torque_joint(self, coriolis_gravity, q_curr: np.array, qd_curr: np.array, desired_pos, desired_ori, tau_last):
        q_target = self.kdl_solver.ikSolver(np.array(desired_pos), np.array(desired_ori).reshape((3, 3)),
                                            q_curr)  # 计算当前关节的目标位置
        M = self.kdl_solver.getInertiaMat(q_curr[:7])  # 机器人的质量矩阵
        ok = False
        tau = tau_last
        if len(q_target) > 0:
            # robot的关节空间控制的计算公式（multiply等同于向量相乘）
            tau = np.multiply(self.kj, q_target - q_curr) - np.multiply(self.Bj, qd_curr)
            tau = np.dot(M, tau)  # 乘上质量矩阵会更稳定一些
            tau += coriolis_gravity[:7]  # 加上科氏力和重力矩
            ok = True  # 标记用于判断解算是否成功
        self.tor = tau
        return self.tor
