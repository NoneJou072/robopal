import numpy as np
from robopal.envs.base import MujocoEnv


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
                 jnt_controller='IMPEDANCE',
                 is_interpolate=False,
                 ):

        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            cam_mode=cam_mode,
            enable_dynamics=jnt_controller is not None
        )

        self.jnt_controller = None
        self.kdl_solver = None
        if jnt_controller == 'IMPEDANCE':
            from robopal.controllers import Jnt_Impedance
            self.jnt_controller = Jnt_Impedance(self.robot)
            self.kdl_solver = self.jnt_controller.kdl_solver
        if jnt_controller is None:
            from robopal.commons.pin_utils import PinSolver
            self.kdl_solver = PinSolver(robot.urdf_path)

        self.interpolator = None
        if is_interpolate:
            self._init_interpolator(self.robot.single_arm)

    def inner_step(self, action):
        if self.interpolator is None:
            q_target, qdot_target = action, np.zeros(self.robot_dof)
        else:
            q_target, qdot_target = self.interpolator.update_state()

        if self.jnt_controller is None:
            for i in range(self.robot.jnt_num):
                self.mj_data.joint(self.robot.single_arm.joint_index[i]).qpos = q_target[i]
        else:
            torque = self.jnt_controller.compute_jnt_torque(
                q_des=q_target,
                v_des=qdot_target,
                q_cur=self.robot.single_arm.arm_qpos,
                v_cur=self.robot.single_arm.arm_qvel,
            )
            # Send torque to simulation
            for i in range(self.robot.jnt_num):
                self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl = torque[i]

    def gripper_ctrl(self, actuator_name: str = None, gripper_action: int = 1):
        """ Gripper control.

        :param actuator_name: Gripper actuator name.
        :param gripper_action: Gripper action, 0 for close, 1 for open.
        """
        self.mj_data.actuator(actuator_name).ctrl = -40 if gripper_action == 0 else 40

    def step(self, action):
        if self.interpolator is not None:
            self.interpolator.update_target_position(action)

        for i in range(int(self.control_timestep / self.model_timestep)):
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))
            super().step(action)

    def _init_interpolator(self, Arm):
        from robopal.commons.interpolators import OTG
        self.interpolator = OTG(
            OTG_dim=self.robot_dof,
            control_cycle=self.control_timestep,
            max_velocity=0.2,
            max_acceleration=0.4,
            max_jerk=0.6
        )
        self.interpolator.set_params(Arm.arm_qpos, Arm.arm_qvel)

    def reset(self):
        super().reset()
        if self.interpolator is not None:
            self.interpolator.set_params(self.robot.single_arm.arm_qpos,
                                         np.zeros(self.robot_dof))


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = JntCtrlEnv(
        robot=DianaMed(),
        renderer='viewer',
        is_render=True,
        control_freq=20,
        is_interpolate=True,
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.33116, -0.39768533, 0.96947228, 0.33116, -0.39768533, 0.66947228, 0])
        env.step(action)
        if env.is_render:
            env.render()
