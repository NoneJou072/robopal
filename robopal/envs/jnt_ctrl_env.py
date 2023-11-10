import numpy as np
from robopal.envs.base import MujocoEnv
from robopal.controllers import controllers


class JntCtrlEnv(MujocoEnv):
    """ Single arm environment.

    :param robot(str): Robot configuration.
    :param is_render: Choose if use the renderer to render the scene or not.
    :param renderer: Choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
    :param jnt_controller: Choose the joint controller.
    :param control_freq: Upper-layer control frequency. i.g. frame per second-fps
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    :param enable_camera_viewer: Use camera or not.
    :param cam_mode: Camera mode, "rgb" or "depth".
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=200,
                 enable_camera_viewer=False,
                 cam_mode='rgb',
                 jnt_controller='JNTIMP',
                 is_interpolate=False,
                 ):

        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            cam_mode=cam_mode,
        )
        self.is_interpolate = is_interpolate
        # choose controller
        if jnt_controller not in controllers:
            raise AttributeError("No joint controller specified, or the controller is not supported.")
        self.jnt_controller = controllers[jnt_controller](
            self.robot,
            is_interpolate=is_interpolate,
            interpolator_config={'dof': self.robot_dof, 'control_timestep': self.control_timestep}
        )

        self.kdl_solver = self.jnt_controller.kdl_solver  # shallow copy

        self.nsubsteps = int(self.control_timestep / self.model_timestep)
        if self.nsubsteps == 0:
            raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                             "Current Model-Timestep:{}".format(self.model_timestep))

    def inner_step(self, action):
        if self.jnt_controller.name == 'JNTNONE':
            qpos = self.jnt_controller.step_controller(action)
            for i in range(self.robot.jnt_num):
                self.mj_data.joint(self.robot.joint_index[i]).qpos = qpos[i]
        else:
            torque = self.jnt_controller.step_controller(action)
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
            self.jnt_controller.step_interpolator(action)
        # step into inner loop
        for i in range(self.nsubsteps):
            super().step(action)

    def reset(self):
        super().reset()
        if self.is_interpolate:
            self.jnt_controller.reset_interpolator(self.robot.arm_qpos, self.robot.arm_qvel)


if __name__ == "__main__":
    from robopal.robots.diana_med import DianaMed

    env = JntCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=20,
        is_interpolate=False,
        jnt_controller='JNTIMP',
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        env.step(action)
        if env.is_render:
            env.render()
