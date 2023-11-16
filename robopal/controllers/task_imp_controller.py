import numpy as np
from robopal.commons.pin_utils import PinSolver
import robopal.commons.transform as trans

def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
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

    w1, w2, w3 = 0.5, 0.5, 0.5
    error = w1 * np.cross(rc1, rd1) + w2 * np.cross(rc2, rd2) + w3 * np.cross(rc3, rd3)

    return error


class CartImpedance:
    """ Cartesian Impedance Controller in the end-effector frame """
    def __init__(
            self,
            robot,
            is_interpolate=False,
            interpolator_config: dict = None
    ):
        self.name = 'CARTIMP'
        self.dofs = 7
        self.robot = robot
        self.kdl_solver = PinSolver(robot.urdf_path)

        # hyper-parameters of impedance
        self.Bc = np.zeros(6)
        self.Kc = np.zeros(6)

        self.set_cart_params(
            b=np.array([200, 800, 800, 800, 800, 800], dtype=np.float32),
            k=np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)
        )

    def set_cart_params(self, b: np.ndarray, k: np.ndarray):
        """set the parameters of the impedance controller in the cartesian space """
        self.Bc = b
        self.Kc = k

    def step_controller(self, desired_pos, desired_ori):
        """ compute the torque in the joint space from the impedance controller in the cartesian space

        desired_pos: desired position
        desired_ori: desired orientation
        """
        desired_ori = trans.quat_2_mat(desired_ori)
        q_curr = self.robot.arm_qpos
        qd_curr = self.robot.arm_qvel

        current_pos, current_ori = self.kdl_solver.fk(q_curr)

        J = self.kdl_solver.get_full_jac(q_curr)
        J_inv = self.kdl_solver.get_full_jac_pinv(q_curr)
        Jd = self.kdl_solver.get_jac_dot(q_curr, qd_curr)

        M = self.kdl_solver.get_inertia_mat(q_curr)
        Md = np.dot(J_inv.T, np.dot(M, J_inv))  # 目标质量矩阵

        pos_error = desired_pos - current_pos  # 位置偏差
        ori_error = orientation_error(desired_ori, current_ori)  # 姿态偏差
        x_error = np.concatenate([pos_error, ori_error])  # 两者拼接
        v_error = -np.dot(J, qd_curr)

        sum = self.Kc * x_error - np.dot(np.dot(Md, Jd), qd_curr) + self.Bc * v_error
        inertial = np.dot(M, J_inv)  # the inertial matrix in the end-effector frame

        C = self.kdl_solver.get_coriolis_mat(q_curr, qd_curr)
        g = self.kdl_solver.get_gravity_mat(q_curr)
        coriolis_gravity = C[-1] + g
        tau = np.dot(inertial, sum) + coriolis_gravity

        return tau
