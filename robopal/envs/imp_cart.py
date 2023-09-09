import numpy as np
from robopal.envs import MujocoEnv
from robopal.commons.pin_utils import PinSolver


class ImpedGymEnv(MujocoEnv):

    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="mujoco_viewer",
                 control_freq=50,
                 ) -> None:
        super(ImpedGymEnv, self).__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
        )
        self.kdl_solver = PinSolver(robot.urdf_path)

        # 导入阻抗模块
        from robopal.commons.controllers.Cart_Impedance import Cart_Impedance
        self.Cart_imped = Cart_Impedance()
        self.Cart_imped.set_cart_Params(
            m=np.diag(0.1 * np.ones(6)),
            b=np.array([120, 120, 120, 70, 70, 70], dtype=np.float32),
            k=np.array([300, 300, 300, 300, 300, 300], dtype=np.float32)
        )

    def preStep(self, action):
        # 根据阻抗控制获取末端输入力矩
        current_pos, current_ori = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)
        tor = self.Cart_imped.torque_cartesian(
            self.mj_data.qfrc_bias[self.robot.single_arm.get_Arm_id()[0]:],
            self.robot.single_arm.arm_qpos,
            self.robot.single_arm.arm_qvel,
            current_pos,
            current_ori,
            desired_pos=np.array([0.686, 0.005, 0.369]),
            desired_ori=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))

        for i in range(7):
            self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl = tor[i]


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = ImpedGymEnv(
        robot=DianaMed(),
        is_render=True,
        renderer="viewer",
        control_freq=200,
    )

    for i in range(int(1e6)):
        action = np.array([0.5, 0, 1])
        env.step(action)
        if env.is_render:
            env.render()
