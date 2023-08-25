import numpy as np
from robopal.envs.base import MujocoEnv
from robopal.utils.KDL_utils import KDL_utils
from robopal.utils.controllers import Jnt_Impedance, Jnt_PD


class SingleArmEnv(MujocoEnv):
    """

    :param xml_path(string): Load xml file from xml_path to build the mujoco model.
    :param is_render: Choose if use the renderer to render the scene or not.
    :param renderer: Choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
    :param control_freq: Upper-layer control frequency.
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 jnt_controller='PD',
                 control_freq=200,
                 is_interpolate=False
                 ):

        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq
        )

        self.kdl_solver = KDL_utils(robot.urdf_path)

        if jnt_controller == 'PD':
            self.jnt_controller = Jnt_PD(self.robot)
        elif jnt_controller == 'IMPEDANCE':
            self.jnt_controller = Jnt_Impedance(self.robot)
        else:
            raise ValueError('Invalid controller name.')

        self.interpolator = None
        if is_interpolate:
            self._init_interpolator(self.robot.single_arm)

    def preStep(self, action):
        if self.interpolator is None:
            q_target, qdot_target = action, np.zeros(self.robot_dof)
        else:
            try:
                q_target, qdot_target = self.interpolator.updateState()
            except ValueError:
                print(action)

        m = self.kdl_solver.getInertiaMat(self.robot.single_arm.arm_qpos)
        c = self.kdl_solver.getCoriolisMat(self.robot.single_arm.arm_qpos, self.robot.single_arm.arm_qvel)
        g = self.kdl_solver.getGravityMat(self.robot.single_arm.arm_qpos)

        torque = self.jnt_controller.compute_jnt_torque(
            q_des=q_target,
            v_des=qdot_target,
            q_cur=self.robot.single_arm.arm_qpos,
            v_cur=self.robot.single_arm.arm_qvel,
            coriolis_gravity=c[-1] + g,
            M=m
        )
        # Send torque to simulation
        for i in range(7):
            self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl = torque[i]

    def step(self, action):
        if self.interpolator is not None:
            self.interpolator.updateInput(action)

        for i in range(int(self.control_timestep / self.model_timestep)):
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))
            super().step(action)

    def _init_interpolator(self, Arm):
        from robopal.utils.interpolators import OTG
        self.interpolator = OTG(
            OTG_Dof=self.robot_dof,
            control_cycle=self.control_timestep,
            max_velocity=0.05,
            max_acceleration=0.1,
            max_jerk=0.2
        )
        self.interpolator.setOTGParam(Arm.arm_qpos, Arm.arm_qvel)

    def reset(self):
        super().reset()
        if self.interpolator is not None:
            self.interpolator.setOTGParam(self.robot.single_arm.arm_qpos,
                                          np.zeros(self.robot_dof))


if __name__ == "__main__":
    from robopal.assets.robots import DianaMed

    env = SingleArmEnv(
        robot=DianaMed(),
        is_render=True,
        renderer='viewer',
        jnt_controller='PD',
        control_freq=200,
        is_interpolate=True
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([0.1, 0.301, 0.2, 2.151, -0.40, -1.281, 0.4])
        env.step(action)
        if env.is_render:
            env.render()
