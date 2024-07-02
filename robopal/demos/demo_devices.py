""" Camera calibration environment.
    In this case, we will show the detail process of hand-eye calibration.
    Press 'Enter' to take a picture.
"""
import numpy as np

from robopal.envs.robot import RobotEnv
from robopal.robots.diana_med import DianaCalib
import robopal.commons.transform as T


if __name__ == "__main__":
    device = input("Please input the device you want to use (keyboard/gamepad): ")
    if device == 'keyboard':
        from robopal.devices import Keyboard
        device = Keyboard()
    elif device == 'gamepad':
        from robopal.devices import Gamepad
        device = Gamepad()
    else:
        raise ValueError("Invalid device type.")

    env = RobotEnv(
        robot=DianaCalib,
        render_mode='human',
        control_freq=200,
        controller='CARTIK',
        is_show_camera_in_cv=True,
        is_render_camera_offscreen=True,
        camera_in_render='cam'
    )

    device.start()
    
    env.reset()
    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])

    for t in range(int(1e6)):
        device_outputs = device.get_outputs()
        action[:3] += device_outputs[0]
        action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(T.euler_2_mat(device_outputs[1])))
        env.step(action)
    env.close()
