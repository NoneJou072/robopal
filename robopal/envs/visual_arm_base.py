import numpy as np
import time
from robopal.envs.mujoco_base import MujocoEnv
from robopal.utils.KDL_utils import KDL_utils


class VisualArmEnv(MujocoEnv):
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
                 renderer="mujoco_viewer",
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
        self.base_time = self.control_timestep

        self.interpolator = None
        if is_interpolate:
            self._initInterpolator(self.robot.single_arm)

    def actuate_base(self, base):
        for i in range(len(self.robot.base_joint_index)):
            acc_desire = 100 * (base - self.mj_data.joint(self.robot.base_joint_index[i]).qpos[0]) - \
                         66 * (self.mj_data.joint(self.robot.base_joint_index[i]).qvel[0])

            tau_target = np.subtract(np.dot(2, acc_desire),
                                     self.mj_data.joint(self.robot.base_joint_index[i]).qfrc_bias[0])
            self.mj_data.actuator('base_stand_joint_motor').ctrl = tau_target

    def actuate_J(self, q_target, qdot_target, Arm):
        """ Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_target: joint position
        :param qdot_target: joint velocity
        """
        acc_desire = [
            (self.robot.kp[i] * (q_target[i] - Arm.arm_qpos[i]) -
             (self.robot.kd[i] * Arm.arm_qvel[i])) for i in range(Arm.jnt_num)]
        qM = self.kdl_solver.getInertiaMat([Arm.arm_qpos[i] for i in range(len(Arm.joint_index))])
        tau_target = np.dot(qM, acc_desire) + np.array(
            [self.mj_data.joint(Arm.joint_index[i]).qfrc_bias[0] for i in range(len(Arm.joint_index))])
        # Send torque to simulation
        for i in range(7):
            self.mj_data.actuator(Arm.actuator_index[i]).ctrl = tau_target[i]

    def preStep(self, action):
        q_target, qdot_target = action, np.zeros(self.robot_dof)
        if self.interpolator is not None:
            try:
                q_target, qdot_target = self.interpolator.updateState()
            except ValueError:
                print(action)
        self.actuate_J(q_target, qdot_target, self.robot.single_arm)

    def step(self, action):
        if self.interpolator is not None:
            self.interpolator.updateInput(action)

        ctrl_cur_time = time.time()
        for i in range(int(self.control_timestep / self.model_timestep)):
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))

            super().step(action)
        self.base_time = time.time() - ctrl_cur_time

    def _initInterpolator(self, Arm):
        from robopal.utils.interpolators import OTG
        self.interpolator = OTG(
            OTG_Dof=self.robot_dof,
            control_cycle=0.0005,
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
    from robopal.assets.robots.diana_med import DianaMed

    env = VisualArmEnv(
        robot=DianaMed(),
        is_render=True,
        control_freq=200,
        is_interpolate=True
    )
    env.reset()
    for t in range(int(1e6)):
        action = np.array([1, -0.501, 0.0, 2.151, 0.00, 1.081, 0.0])
        env.step(action)
        if env.is_render:
            env.render()
