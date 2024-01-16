import numpy as np
from robopal.commons.pin_utils import PinSolver


class JntImpedance(object):
    def __init__(
            self,
            robot,
            is_interpolate=False,
            interpolator_config: dict = None
    ):
        self.name = 'JNTIMP'
        self.dofs = robot.jnt_num
        self.robot = robot
        self.kdl_solver = PinSolver(robot.urdf_path)

        # hyperparameters of impedance controller
        self.Bj = np.zeros(self.dofs)
        self.kj = np.zeros(self.dofs)

        self.set_jnt_params(
            b=600.0 * np.ones(self.dofs),
            k=1000.0 * np.ones(self.dofs),
        )

        # choose interpolator
        self.interpolator = None
        if is_interpolate:
            interpolator_config.setdefault('init_qpos', robot.arm_qpos)
            interpolator_config.setdefault('init_qvel', robot.arm_qvel)
            self._init_interpolator(interpolator_config)

    def set_jnt_params(self, b: np.ndarray, k: np.ndarray):
        """ Used for changing the parameters. """
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
        if self.interpolator is not None:
            q_des, v_des = self.interpolator.update_state()

        M = self.kdl_solver.get_inertia_mat(q_cur)
        C = self.kdl_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kdl_solver.get_gravity_mat(q_cur)
        coriolis_gravity = C[-1] + g

        acc_desire = self.kj * (q_des - q_cur) + self.Bj * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau

    def step_controller(self, action):
        q_target, qdot_target = action, np.zeros(self.dofs)

        torque = self.compute_jnt_torque(
            q_des=q_target,
            v_des=qdot_target,
            q_cur=self.robot.arm_qpos,
            v_cur=self.robot.arm_qvel,
        )
        return torque

    def _init_interpolator(self, cfg: dict):
        try:
            from robopal.controllers.interpolators import OTG
        except ImportError:
            raise ImportError("Please install ruckig first: pip install ruckig")
        self.interpolator = OTG(
            OTG_dim=cfg['dof'],
            control_cycle=cfg['control_timestep'],
            max_velocity=0.2,
            max_acceleration=0.4,
            max_jerk=0.6
        )
        self.interpolator.set_params(cfg['init_qpos'], cfg['init_qvel'])

    def step_interpolator(self, action):
        self.interpolator.update_target_position(action)

    def reset_interpolator(self, arm_qpos, arm_qvel):
        self.interpolator.set_params(arm_qpos, arm_qvel)
