import numpy as np

from robopal.controllers.base_controller import BaseController


class JointTorqueController(BaseController):
    def __init__(
            self,
            robot,
            is_interpolate=False,
            interpolator_config: dict = None
    ):
        super().__init__(robot)

        self.name = 'JNTTAU'

        # choose interpolator
        self.interpolator = None
        if is_interpolate:
            interpolator_config.setdefault('init_qpos', robot.get_arm_qpos())
            interpolator_config.setdefault('init_qvel', robot.get_arm_qvel())
            self._init_interpolator(interpolator_config)

    def step_controller(self, action):
        """ 

        :param action: joint torque
        :return: joint torque
        """
        if isinstance(action, np.ndarray):
            action = {self.robot.agents[0]: action}

        return action

    def _init_interpolator(self, cfg: dict):
        try:
            from robopal.controllers.planners.interpolators import OTG
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

    def reset(self):
        if self.interpolator is not None:
            self.interpolator.set_params(self.robot.get_arm_qpos(), self.robot.get_arm_qvel())
