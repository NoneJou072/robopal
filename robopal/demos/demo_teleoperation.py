import numpy as np
import logging

from robopal.robots.diana_med import DianaGrasp
from robopal.envs import RobotEnv
from robopal.plugins.devices.keyboard import KeyboardIO
import robopal.commons.transform as T

logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":
    keyboard_recoder = KeyboardIO()

    env = RobotEnv(
        robot=DianaGrasp,
        render_mode='human',
        control_freq=100,
        is_interpolate=False,
        controller='CARTIK',
    )

    action = np.array([0.49, 0.134, 0.49, 1, 0, 0, 0])

    env.reset()
    for t in range(int(2e4)):
        action[:3] += keyboard_recoder.get_end_pos_offset()
        action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(keyboard_recoder.get_end_rot_offset()))
        env.step(action)
    env.close()
