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
        self.timer = 0
        self.init_pos, self.init_rot = self.kdl_solver.getEeCurrentPose(self.robot.single_arm.arm_qpos)
        self.q_curr = np.zeros(7)
        self.qd_curr = np.zeros(7)
        self.desire_ee_force = np.zeros(3)
        self.current_ee_force = np.zeros(3)
        self.x_pos = np.zeros(3)
        self.x_ori = np.zeros(9).reshape([3, 3])
        self.goal_pos = np.zeros(3)
        self.tor = np.zeros(7)
        self.tau_last = np.zeros(7)
        self.h_e = np.zeros(6)

        self.action_dim = (4,)
        self.obs_dim = (6, 3)
        self.action_low = np.array(4)
        self.action_high = np.array(4)

        # 导入阻抗模块
        from robopal.utils.controllers.Cart_Impedance import Cart_Impedance
        self.Cart_imped = Cart_Impedance()
        self.mc, self.kc, self.dc = np.array([0] * 6, dtype=np.float32), \
                                    np.array([300, 300, 300, 300, 300, 300], dtype=np.float32), \
                                    np.array([120, 120, 120, 70, 70, 70], dtype=np.float32)
        self.Cart_imped.set_cart_Params(
            m=np.diag(np.concatenate((0.1 * np.ones(3), 0.1 * np.ones(3)), axis=0)),
            b=self.dc,
            k=self.kc
        )

    def step(self, action):
        self.timer += 1

        # if 1000 <= self.timer < 2200:
        #     self.mj_data.xfrc_applied[7][0] = 30  # 表示连杆所受外力的力矩，第一维是每个body的ID，第二维有六个量（x，y，z方向的受力，后边是力矩）
        # else:
        #     self.mj_data.xfrc_applied[7][0] = 0  # 还原数值，不施加外力

        if self.timer in range(0, 650):
            self.goal_pos = np.array([0.94, 0.005, 0.38])
        elif self.timer in range(650, 1400):
            self.goal_pos = np.array([0.94, -0.04, 0.38])
        elif self.timer in range(1400, 2000):
            self.goal_pos = np.array([0.94, 0.04, 0.31])

            # 重设imped参数的参考值
        self.Cart_imped.set_cart_Params(self.mc, self.dc, self.kc)  # cart space
        # 状态
        self.q_curr, self.qd_curr, self.x_pos, self.x_ori, self.var_pos, self.var_h_e = self._get_state()
        # 变阻抗参数
        # self.imped.K = self.imped.adapted_K(h_e=self.h_e)
        # self.imped.B = self.imped.adapted_B(vel=self.curr_vel)
        # 将关节扭矩下发至仿真
        super().step(action)
        _ = None
        done = True
        return self._get_obs(), self.reward(action), done, _

    def preStep(self, action):
        # 根据阻抗控制获取末端输入力矩
        self.tau_last = self.tor
        self.tor = self.Cart_imped.torque_cartesian(
            self.mj_data.qfrc_bias[self.robot.single_arm.get_Arm_id()[0]:],
            self.robot.single_arm.arm_qpos,
            self.robot.single_arm.arm_qvel,
            self.mj_data.site_xpos[0],
            self.mj_data.site_xmat[0].reshape([3, 3]),
            self.goal_pos,
            np.array([0, 0, 1, 0, 1, 0, -1, 0, 0]))
        print(self.mj_data.site_xpos[0])

        # print(self.mj_data.qfrc_bias[self.robot.single_arm.get_Arm_id()[0]:self.robot.single_arm.get_Arm_id()[6]+1])
        for i in range(7):
            self.mj_data.actuator(self.robot.single_arm.actuator_index[i]).ctrl = self.tor[i]

    def reset(self):
        self.timer = 0
        self.goal_pos = np.array([0.986, 0.005, 0.369])
        self.desire_ee_force[2] = 1.2
        self.last_h_e = np.zeros(6)
        super().reset()
        return self._get_obs()

    def _get_state(self):
        q_curr = self.robot.single_arm.arm_qpos
        qd_curr = self.robot.single_arm.arm_qvel
        x_pos = self.mj_data.site_xpos[0]
        x_ori = self.mj_data.site_xmat[0].reshape([3, 3])
        var_pos = self.goal_pos - x_pos
        var_h_e = self.tor[:7] - self.tau_last
        self.tau_last = self.tor
        return [q_curr, qd_curr, x_pos, x_ori, var_pos, var_h_e]

    def reward(self, action):
        reward = 0.0
        print(f"reward:{reward}")
        return reward

    def _get_obs(self):
        obs = np.zeros(shape=self.obs_dim, dtype=np.float32)
        obs[0] = self.current_ee_force
        obs[1] = self.x_pos
        obs[2] = self.var_h_e[:3]
        return obs

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

    for i in range(int(1e6)):
        action = np.array([1, 0, 1])
        obs, reward, done, _ = env.step(action)
        if env.is_render:
            env.render()
