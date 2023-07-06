import numpy as np
from robopal.envs import SingleArmEnv


class ImpedGymEnv(SingleArmEnv):
    """
        This class is only for impedance controlling without cosserat.
    """

    def __init__(self,                  
                 robot=None,
                 is_render=True,
                 renderer="mujoco_viewer",
                 control_freq=50,
                 is_interpolate=True,
                 ) -> None:
        super(ImpedGymEnv, self).__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate,
        )
        # 全局属性
        self.q_curr = self.robot.single_arm.arm_qpos
        self.qd_curr = self.robot.single_arm.arm_qvel
        self.x_pos = self.mj_data.site_xpos[0]
        self.x_ori = self.mj_data.site_xmat[0].reshape([3, 3])
        self.mj, self.kj, self.dj = np.array([0] * 7, dtype=np.float32), np.array([100] * 7,
                                                                                  dtype=np.float32), np.array([30] * 7,
                                                                                                              dtype=np.float32)  # 关节空间的参数
        self.tau_last = np.zeros(7)
        self.tor = np.zeros(7)

        # 导入弹性杆模块
        from robopal.utils.plugins.cosserat.scripts.ElasticaModel import ElasticaModel
        self.ela_model = ElasticaModel()
        self.ela_model.load()

        # 导入阻抗模块
        from robopal.utils.controllers.Jnt_Impedance import Jnt_Impedance
        self.Jnt_imped = Jnt_Impedance()
        self.Jnt_imped.set_jnt_Params(
            m=np.diag(np.concatenate((0.1 * np.ones(3), 0.1 * np.ones(3)), axis=0)),
            b=self.dj,
            k=self.kj
        )

        # 末端执行器初始位置
        self.init_pos, self.init_rot = self.kdl_solver.getEeCurrentPose(self.mj_data.qpos[:7])

    def step(self, action):

        if 1000 <= self.cur_time < 2200:
            self.mj_data.xfrc_applied[7][0] = 30  # 表示连杆所受外力的力矩，第一维是每个body的ID，第二维有六个量（x，y，z方向的受力，后边是力矩）
        else:
            self.mj_data.xfrc_applied[7][0] = 0  # 还原数值，不施加外力

        # 重设imped参数的参考值
        self.Jnt_imped.set_jnt_Params(self.mj, self.dj, self.kj)  # joint space

        # 状态
        self.q_curr = self.robot.single_arm.arm_qpos
        self.qd_curr = self.robot.single_arm.arm_qvel
        self.x_pos = self.mj_data.site_xpos[0]
        self.x_ori = self.mj_data.site_xmat[0].reshape([3, 3])

        # 变阻抗参数
        # self.imped.K = self.imped.adapted_K(h_e=self.h_e)
        # self.imped.B = self.imped.adapted_B(vel=self.curr_vel)

        # 将关节扭矩下发至仿真
        super().step(action)

        if self.is_render:
            self.render()
        _ = None
        done = False
        return done, _

    def preStep(self, action):
        # 根据阻抗控制获取末端输入力矩
        self.tor = self.Jnt_imped.torque_joint(
            self.mj_data.qfrc_bias[self.robot.single_arm.get_Arm_id()[0]:],
            self.robot.single_arm.arm_qpos,
            self.robot.single_arm.arm_qvel,
            action,
            np.array([0, 0, -1, 0, 1, 0, 1, 0, 0]),
            self.tau_last)
        
        for i in range(7):
            # print(self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl)
            self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl = self.tor[i]

    def render(self, mode="human"):
        super().render()


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = ImpedGymEnv(
        robot=DianaMed(),
        is_render=True,
        renderer="mujoco_viewer",
        control_freq=200,
        is_interpolate=True,
    )

    for i in range(30000):
        action = np.array([0.5, 0, 0.5])
        done, _ = env.step(action)
