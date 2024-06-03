""" Camera calibration environment.
    In this case, we will show the detail process of hand-eye calibration.
    Press 'Enter' to take a picture.
"""
import numpy as np

from robopal.envs.robot import RobotEnv
from robopal.robots.diana_med import DianaCalib
from robopal.plugins.devices.keyboard import KeyboardIO
import robopal.commons.transform as T


if __name__ == "__main__":

    env = RobotEnv(
        robot=DianaCalib,
        render_mode='human',
        control_freq=200,
        controller='CARTIK',
        is_show_camera_in_cv=True,
        camera_in_render='cam'
    )

    keyboard_recoder = KeyboardIO()
    keyboard_recoder.start()
    
    env.reset()
    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])

    for t in range(int(1e6)):
        action[:3] += keyboard_recoder.get_end_pos_offset()
        action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(keyboard_recoder.get_end_rot_offset()))
        env.step(action)
    env.close()
