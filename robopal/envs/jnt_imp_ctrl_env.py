import numpy as np
from robopal.envs.base import MujocoEnv


class JntCtrlEnv(MujocoEnv):
    """ Single arm environment.

    :param robot(str): Robot configuration.
    :param is_render: Choose if use the renderer to render the scene or not.
    :param renderer: Choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
    :param jnt_controller: Choose the joint controller.
    :param control_freq: Upper-layer control frequency.
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    :param is_camera_used: Use camera or not.
    :param cam_mode: Camera mode, "rgb" or "depth".
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 jnt_controller='IMPEDANCE',
                 control_freq=200,
                 is_interpolate=False,
                 is_camera_used=False,
                 cam_mode='rgb'
                 ):

        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_camera_used=is_camera_used,
            cam_mode=cam_mode
        )

        if jnt_controller == 'IMPEDANCE':
            from robopal.controllers import Jnt_Impedance
            self.jnt_controller = Jnt_Impedance(self.robot)
        else:
            raise ValueError('Invalid controller name.')

        self.interpolator = None
        if is_interpolate:
            self._init_interpolator(self.robot.single_arm)

    @property
    def kdl_solver(self):
        return self.jnt_controller.kdl_solver

    def preStep(self, action):
        if self.interpolator is None:
            q_target, qdot_target = action, np.zeros(self.robot_dof)
        else:
            q_target, qdot_target = self.interpolator.update_state()

        torque = self.jnt_controller.compute_jnt_torque(
            q_des=q_target,
            v_des=qdot_target,
            q_cur=self.robot.single_arm.arm_qpos,
            v_cur=self.robot.single_arm.arm_qvel,
        )
        # Send torque to simulation
        for i in range(7):
            self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl = torque[i]

    def step(self, action):
        if self.interpolator is not None:
            self.interpolator.update_input(action)

        for i in range(int(self.control_timestep / self.model_timestep)):
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))
            super().step(action)

    def _init_interpolator(self, Arm):
        from robopal.commons.interpolators import OTG
        self.interpolator = OTG(
            OTG_Dof=self.robot_dof,
            control_cycle=self.control_timestep,
            max_velocity=0.05,
            max_acceleration=0.1,
            max_jerk=0.2
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
        control_freq=200,
        is_interpolate=False,
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.33116, -0.39768533, 0.96947228, 0.33116, -0.39768533, 0.66947228, 0])
        env.step(action)
        if env.is_render:
            env.render()
