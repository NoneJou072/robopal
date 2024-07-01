import numpy as np
from robopal.demos.demo_admit_rl.demo_admit_test import AdmitGymEnv
from robopal.undo.unity.mjremote import mjremote


class VisualForceAdmit(AdmitGymEnv):
    """
        UnityForceVisual 以 mujoco 为后端，unity 为前端，基于导纳控制算法，将所受外力
        可视化，用于效果展示。
    """
    def __init__(self,
                 robot=None,
                 render_mode="human",
                 controller=1,
                 control_freq=50,
                 is_interpolate=True,
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            controller=controller,
            control_freq=control_freq,
            is_interpolate=is_interpolate,
        )

    def step(self, action):
        if self.timer in range(70, 100):
            self.current_ee_force = np.array([2, 2, 2])
        elif self.timer in range(100, 200):
            self.current_ee_force = self.current_ee_force + 0.001
        elif self.timer in range(200, 300):
            self.current_ee_force[0] = self.current_ee_force[0] - 0.02
            self.current_ee_force[1] = self.current_ee_force[1] - 0.02
        else:
            self.current_ee_force = np.zeros(3)
        return super(VisualForceAdmit, self).step(action)


if __name__ == "__main__":
    from robopal.robots.diana_med import DianaMed

    env = VisualForceAdmit(
        robot=DianaMed,
        render_mode="human",
        controller=1,
        control_freq=100,
        is_interpolate=True,
    )
    env.reset()
    m = mjremote()
    print('Connect: ', m.connect())
    for i in range(int(1e6)):
        action = np.array([0.4, 0.4, 0.4, 2])
        obs, reward, done, _ = env.step(action)
        m.setqpos(env.mj_data.qpos[:env.robot_freedom])
        m.sendForce(env.current_ee_force)
        env.render()
        if i % 800 == 0:
            env.reset()
