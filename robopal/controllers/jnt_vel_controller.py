import numpy as np
from robopal.commons.pin_utils import PinSolver
from collections import deque


class JntVelController:
    def __init__(
            self,
            robot,
            is_interpolate=False,
            interpolator_config: dict = None
    ):
        self.name = 'JNTVEL'
        self.kdl_solver = PinSolver(robot.urdf_path)

        # hyperparameters of impedance controller
        self.k_p = np.zeros(7)
        self.k_d = np.zeros(7)

        self.set_jnt_params(
            p=3.0 * np.ones(7),
            d=0.003 * np.ones(7),
        )

        self.last_err = np.zeros(robot.single_arm.jnt_num)
        self.err_buffer = deque(maxlen=5)

        print("Jnt_Impedance Initialized!")

    def set_jnt_params(self, p: np.ndarray, d: np.ndarray):
        self.k_p = p
        self.k_d = d

    def compute_jnt_torque(
            self,
            q_des: np.ndarray,
            v_des: np.ndarray,
            q_cur: np.ndarray,
            v_cur: np.array,
    ):
        """ robot的关节空间控制的计算公式
            Compute desired torque with robot dynamics modeling:
            > k_p * (vd - v) = tau

        :param q_des: desired joint position
        :param v_des: desired joint velocity
        :param q_cur: current joint position
        :param v_cur: current joint velocity
        :return: desired joint torque
        """
        C = self.kdl_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kdl_solver.get_gravity_mat(q_cur)
        coriolis_gravity = C[-1] + g

        err = v_des - v_cur
        derr = err - self.last_err
        self.last_err = err
        self.err_buffer.append(derr)
        tau = self.k_p * err - self.k_d * np.asarray(self.err_buffer).flatten().mean() + coriolis_gravity

        return tau
