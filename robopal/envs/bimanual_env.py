import numpy as np
import time
from robopal.envs.base import MujocoEnv
from robopal.utils.pin_utils import PinSolver
from robopal.utils.interpolators import OTG


class DoubleArmEnv(MujocoEnv):
    """

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
                 control_freq=200,
                 is_interpolate=False
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq
        )
        self.left_arm = self.robot.left_arm
        self.right_arm = self.robot.right_arm
        self.kdl_solver = PinSolver(robot.urdf_path)
        self.base_time = self.control_timestep

        if is_interpolate:
            self._initInterpolator(self.left_arm)
            self._initInterpolator(self.right_arm)

    def action_size(self):
        return self.robot.jnt_num

    def actuate_base(self, base):
        for i in range(len(self.robot.base_joint_index)):
            acc_desire = 100 * (base - self.mj_data.joint(self.robot.base_joint_index[i]).qpos[0]) - \
                         66 * (self.mj_data.joint(self.robot.base_joint_index[i]).qvel[0])
            tau_target = np.subtract(np.dot(2, acc_desire),
                                     self.mj_data.joint(self.robot.base_joint_index[i]).qfrc_bias[0])
            self.mj_data.actuator(self.robot.base_actuator_index[i]).ctrl = tau_target

    def actuate_J(self, q_target, qdot_target, Arm):
        """ Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_target: joint position
        :param qdot_target: joint velocity
        """
        acc_desire = [self.robot.kp[i] * (q_target[i] - Arm.arm_qpos[i]) -
                      self.robot.kd[i] * Arm.arm_qvel[i] for i in range(len(Arm.joint_index))]
        qM = self.kdl_solver.getInertiaMat(Arm.arm_qpos)
        tau_target = np.dot(qM, acc_desire) + np.array([self.mj_data.joint(Arm.joint_index[i]).qfrc_bias[0] for i in
                                                        range(len(Arm.joint_index))])
        # Send torque to simulation
        for i in range(len(Arm.actuator_index)):
            self.mj_data.actuator(Arm.actuator_index[i]).ctrl = tau_target[i]

    def preStep(self, action):
        q_target_l, qdot_target_l = action[1:7], np.zeros(len(self.robot.left_arm.get_Arm_index()))
        q_target_r, qdot_target_r = action[8:14], np.zeros(len(self.robot.right_arm.get_Arm_index()))

        if self.left_arm.interpolator and self.right_arm.interpolator is not None:
            q_target_l, qdot_target_l = self.left_arm.interpolator.updateState()
            q_target_r, qdot_target_r = self.right_arm.interpolator.updateState()

        self.actuate_J(q_target_l, qdot_target_l, self.left_arm)
        self.actuate_J(q_target_r, qdot_target_r, self.right_arm)
        if self.robot.base_joint_index == 0:
            pass
        else:
            self.actuate_base(action[0])

    def step(self, action):
        if self.left_arm.interpolator and self.right_arm.interpolator is not None:
            self.left_arm.interpolator.updateInput(action[1:7])
            self.right_arm.interpolator.updateInput(action[8:14])

        ctrl_cur_time = time.time()
        for i in range(int(self.control_timestep / self.model_timestep)):
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))
            super().step(action)

        self.base_time = time.time() - ctrl_cur_time

    def _initInterpolator(self, Arm):
        Arm.interpolator = OTG(
            OTG_Dof=7,
            control_cycle=0.0005,
            max_velocity=0.05,
            max_acceleration=0.1,
            max_jerk=0.2
        )
        Arm.interpolator.setOTGParam(Arm.arm_qpos, Arm.arm_qvel)


if __name__ == "__main__":
    from robopal.assets.robots.bidiana_med import BiDianaMed

    env = DoubleArmEnv(
        robot=BiDianaMed(),
        is_render=True,
        control_freq=200,
        is_interpolate=True
    )
    action_base = np.array([-0.785])
    action_L = np.array([-1.5, -1.501, 0.731, 1.8, 1.00, 1.081, 0.0])
    action_R = np.array([-1.5, -1.501, 0.731, 1.8, 1.00, 1.081, 0.0])
    for t in range(int(1e6)):
        action = np.concatenate((action_base, action_L, action_R), axis=0)
        env.step(action)
        if env.is_render:
            env.render()
