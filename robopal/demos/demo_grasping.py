import numpy as np

from robopal.envs import PosCtrlEnv
from robopal.robots.diana_med import DianaGrasp


class GraspingEnv(PosCtrlEnv):

    def __init__(self,
                 robot=DianaGrasp(),
                 render_mode='human',
                 control_freq=10,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 is_pd=False,
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
            is_interpolate=is_interpolate,
            is_pd=is_pd,
        )


if __name__ == "__main__":
    env = GraspingEnv()
    env.reset()
    action = env.get_body_pos('green_block') - np.array([0.0, 0.0, 0.32])

    for t in range(int(50)):
        env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = 0.03
        env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = 0.03
        env.step(action)
    for t in range(int(50)):
        env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = -0.02
        env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = -0.02
        env.step(action)
    action += np.array([0.0, 0.0, 0.32])
    for t in range(int(1e5)):
        env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = -0.02
        env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = -0.02
        env.step(action)
    env.close()
