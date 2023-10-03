import numpy as np
from robopal.commons.pin_utils import PinSolver


class JntNone(object):
    def __init__(
            self,
            robot,
            is_interpolate=False,
            interpolator_config: dict = None
    ):
        self.name = 'JNTNONE'
        self.kdl_solver = PinSolver(robot.urdf_path)

        # choose interpolator
        self.interpolator = None
        if is_interpolate:
            interpolator_config.setdefault('init_qpos', robot.single_arm.arm_qpos)
            interpolator_config.setdefault('init_qvel', robot.single_arm.arm_qvel)
            self._init_interpolator(interpolator_config)

        print("No controller has Initialized!")

    def compute_jnt_pos(
            self,
            q_des: np.ndarray,
    ):
        """ Compute the desired joint position and velocity.
        :param q_des: desired joint position
        :return: q_des
        """
        if self.interpolator is not None:
            q_des, _ = self.interpolator.update_state()
        return q_des

    def _init_interpolator(self, cfg: dict):
        from robopal.controllers.interpolators import OTG
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
