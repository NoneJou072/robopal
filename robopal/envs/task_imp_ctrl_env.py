import numpy as np
from robopal.envs import MujocoEnv
from robopal.controllers import Cart_Impedance


class PosCtrlEnv(MujocoEnv):

    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=50,
                 ) -> None:
        super(PosCtrlEnv, self).__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
        )

        self.Cart_imped = Cart_Impedance(self.robot)

    def preStep(self, action):
        # 根据阻抗控制获取末端输入力矩
        tor = self.Cart_imped.step_controller(
            self.robot.single_arm.arm_qpos,
            self.robot.single_arm.arm_qvel,
            desired_pos=action,
            desired_ori=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )

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
        action = np.array([0.686, 0.005, 0.369])
        env.step(action)
        if env.is_render:
            env.render()
