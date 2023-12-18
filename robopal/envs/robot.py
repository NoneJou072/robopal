import numpy as np

from robopal.envs.base import MujocoEnv
from robopal.controllers import controllers


class RobotEnv(MujocoEnv):
    """ Robot environment.

    :param robot: Robot configuration.
    :param render_mode: Choose the render mode.
    :param controller: Choose the controller.
    :param control_freq: Upper-layer control frequency. i.g. frame per second-fps
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    :param enable_camera_viewer: Use camera or not.
    """

    def __init__(self,
                 robot=None,
                 control_freq=200,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 camera_name=None,
                 render_mode='human',
                 ):

        super().__init__(
            robot=robot,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            camera_name=camera_name,
            render_mode=render_mode,
        )
        self.is_interpolate = is_interpolate

        # choose controller
        assert controller in controllers, f"Not supported controller, you can choose from {controllers.keys()}"
        self.controller = controllers[controller](
            self.robot,
            is_interpolate=is_interpolate,
            interpolator_config={'dof': self.robot.jnt_num, 'control_timestep': self.control_timestep}
        )

        self.kdl_solver = self.controller.kdl_solver  # shallow copy

        self.n_substeps = int(self.control_timestep / self.model_timestep)
        if self.n_substeps == 0:
            raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                             "Current Model-Timestep:{}".format(self.model_timestep))

    def inner_step(self, action):
        if self.controller.name == 'JNTNONE':
            qpos = self.controller.step_controller(action)
            for i in range(self.robot.jnt_num):
                self.mj_data.joint(self.robot.joint_index[i]).qpos = qpos[i]
        else:
            torque = self.controller.step_controller(action)
            # Send torque to simulation
            for i in range(self.robot.jnt_num):
                self.mj_data.actuator(self.robot.actuator_index[i]).ctrl = torque[i]

    def gripper_ctrl(self, actuator_name: str = None, gripper_action: int = 1):
        """ Gripper control.

        :param actuator_name: Gripper actuator name.
        :param gripper_action: Gripper action, 0 for close, 1 for open.
        """
        self.mj_data.actuator(actuator_name).ctrl = -40 if gripper_action == 0 else 40

    def step(self, action):
        if self.is_interpolate:
            self.controller.step_interpolator(action)
        # high-level control
        for i in range(self.n_substeps):
            # low-level control
            super().step(action)

    def reset(self):
        if self.is_interpolate:
            self.controller.reset_interpolator(self.robot.arm_qpos, self.robot.arm_qvel)
        super().reset()

    @property
    def dt(self):
        "Time of each upper step in the environment."
        return self.n_substeps * self.mj_model.opt.timestep
