import numpy as np
from robopal.envs import MujocoEnv
from robopal.controllers import CartImpedance


class PosCtrlEnv(MujocoEnv):

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 jnt_controller='IMPEDANCE',
                 control_freq=200,
                 is_interpolate=False,
                 enable_camera_viewer=False,
                 cam_mode='rgb'
                 ) -> None:
        super(PosCtrlEnv, self).__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            cam_mode=cam_mode
        )

        self.Cart_imped = CartImpedance(self.robot)

        # self.interpolator = None
        # if is_interpolate:
        #     self._init_interpolator(self.robot)

    def inner_step(self, action):
        # 根据阻抗控制获取末端输入力矩
        tor = self.Cart_imped.step_controller(
            action,
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            q_curr=self.robot.arm_qpos,
            qd_curr=self.robot.arm_qvel,
        )

        for i in range(7):
            self.mj_data.actuator(self.robot.actuator_index[i]).ctrl = tor[i]

    def step(self, action):
        for i in range(int(self.control_timestep / self.model_timestep)):
            if int(self.control_timestep / self.model_timestep) == 0:
                raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                                 "Current Model-Timestep:{}".format(self.model_timestep))
            super().step(action)


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = PosCtrlEnv(
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
