import numpy as np
from os.path import join, dirname
import os
from robopal.utils.KDL_utils import KDL_utils
work_space_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Cart_Impedance(object):
    def __init__(self):
        urdf_path = join(dirname(__file__), "../../assets/models/manipulators/DianaMed/DianaMed.urdf")
        self.kdl_solver = KDL_utils(urdf_path)
        # hyper-parameters of impedance
        self.Mc = np.zeros(6)
        self.Bc = np.zeros(6)
        self.Kc = np.zeros(6)

        self.init_qpos = [0, -1.6, 1.6, -1.6, -1.6, 0, 0]
        self.joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']

        print("Cart_Impedance Initialized!")

    def set_cart_Params(self, m: np.ndarray, b: np.ndarray, k: np.ndarray):
        """

        Args:
            m:
            b:
            k:

        Returns:

        """
        self.Mc = m
        self.Bc = b
        self.Kc = k

    # desired_pos:期望的位置  desired_ori:期望的姿态  tau_last：传入一个力矩
    def torque_cartesian(self, coriolis_gravity, q_curr, qd_curr, x_pos: np.array, x_ori: np.array, desired_pos,
                         desired_ori):
        M = self.kdl_solver.getInertiaMat(q_curr[:7])
        J = np.diag(np.ones(7))
        try:
            J = self.kdl_solver.getJac(self.kdl_solver.setJntArray(q_curr[:7]))
        except:
            pass
        J_inv = np.linalg.pinv(self.kdl_solver.setNumpyMat(J))  # 雅各比矩阵的逆
        Jd = self.kdl_solver.jacobian_dot(q_curr[:7], qd_curr[:7])  # 雅各比矩阵的微分
        Md = np.dot(J_inv.T, np.dot(M, J_inv))  # 目标质量矩阵
        print(Md)
        tau = coriolis_gravity[:7].copy()

        # 获取末端的位置/姿态/速度/
        x_vel = np.dot(self.kdl_solver.setNumpyMat(J), qd_curr[:7])
        x_pos_vel = np.array([x_vel[0], x_vel[1], x_vel[2]])
        x_ori_vel = np.array([x_vel[3], x_vel[4], x_vel[5]])
        coef = np.dot(M, J_inv)
        xd_error = np.concatenate([-x_pos_vel, -x_ori_vel])  # 末端姿态和位置的拼接
        sum = np.multiply(self.Bc, xd_error)
        pos_error = desired_pos - x_pos  # 位置偏差
        ori_error = self.orientation_error(desired_ori.reshape([3, 3]), x_ori)  # 姿态偏差
        x_error = np.concatenate([pos_error, ori_error])  # 两者拼接

        sum += np.multiply(self.Kc, x_error)
        sum -= np.dot(np.dot(Md, Jd), qd_curr)
        tau += np.dot(coef, sum)
        self.tor = tau
        return self.tor

    def orientation_error(self, desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        """computer ori error from ori to cartesian 姿态矩阵的偏差3*3的
        Args:
            desired (np.ndarray): desired orientation
            current (np.ndarray): current orientation

        Returns:
            _type_: orientation error(from pose(3*3) to eulor angular(3*1))
        """
        rc1 = current[:, 0]
        rc2 = current[:, 1]
        rc3 = current[:, 2]
        rd1 = desired[:, 0]
        rd2 = desired[:, 1]
        rd3 = desired[:, 2]
        if (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3)).all() <= 0.0001:
            w1, w2, w3 = 0.5, 0.5, 0.5
        else:
            w1, w2, w3 = 0.9, 0.5, 0.3

        error = w1 * np.cross(rc1, rd1) + w2 * np.cross(rc2, rd2) + w3 * np.cross(rc3, rd3)

        return error
