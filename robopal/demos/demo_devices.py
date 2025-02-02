import numpy as np
import logging
import robopal
from robopal.envs.robot import RobotEnv
from robopal.robots.diana_med import DianaCalib
import robopal.commons.transform as T


def single_env_test(device):
    """ Camera calibration environment.
    In this case, we will show the detail process of hand-eye calibration.
    Press 'Enter' to take a picture.
    """
    env = RobotEnv(
        robot=DianaCalib,
        render_mode='human',
        control_freq=40,
        controller='CARTIK',
        is_show_camera_in_cv=True,
        is_render_camera_offscreen=True,
        camera_in_render='cam'
    )

    device = device()
    device.start()
    
    env.reset()
    init_pos = env.robot.get_end_xpos()
    init_quat = env.robot.get_end_xquat()
    action = np.concatenate([init_pos, init_quat])

    for t in range(int(1e6)):
        device_outputs = device.get_outputs()
        action[:3] += device_outputs[0]
        action[3:] = T.mat_2_quat(T.quat_2_mat(action[3:]).dot(device_outputs[1]))
        env.step(action)

        if device._reset_flag:
            env.reset()
            action = np.concatenate([init_pos, init_quat])
            device._reset_flag = False
    env.close()


def multi_env_test(device):
    env = robopal.make("BimanualTransport-v0", render_mode='human')

    device = device()
    device.start()
    
    env.reset()

    a1_action = np.array([0, 0, 0, 1], dtype=np.float32)
    a2_action = np.array([0, 0, 0, 1], dtype=np.float32)
    actions = {env.agents[0]: a1_action,
                env.agents[1]: a2_action}
    
    last_agent_id = device._agent_id
    for t in range(int(1e6)):
        device_outputs = device.get_outputs()
        if device._agent_id != last_agent_id:
            if (int(device._gripper_flag) * 2 - 1) != int(actions[env.agents[device._agent_id]][3]):
                device._gripper_flag = not device._gripper_flag
        last_agent_id = device._agent_id

        if device._agent_id == 0:
            a1_action[:3] = device_outputs[0] * 2
            a1_action[3] = int(device._gripper_flag) * 2 - 1
            a2_action[:3] = np.zeros(3)
        elif device._agent_id == 1:
            a1_action[:3] = np.zeros(3)
            a2_action[:3] = device_outputs[0] * 2
            a2_action[3] = int(device._gripper_flag) * 2 - 1
        else:
            raise ValueError("Invalid agent id.")
        
        actions = {env.agents[0]: a1_action,
                    env.agents[1]: a2_action}

        _, _, _, _, info = env.step(actions)

        if device._reset_flag or info["agent0"]["is_success"]:
            env.reset()
            a1_action = np.zeros(4)
            a2_action = np.zeros(4)
            device._reset_flag = False
    env.close()


if __name__ == "__main__":
    device = input("Please input the device you want to use (keyboard/gamepad): ")
    if device == 'keyboard':
        from robopal.devices import Keyboard
        device = Keyboard
    elif device == 'gamepad':
        from robopal.devices import Gamepad
        device = Gamepad
    else:
        raise ValueError("Invalid device type.")

    # switch environment
    logging.info("Environments list:\n 1. [Single Agent]CameraCalibration \n 2. [Multi Agent]BimanualTransport")
    env_id = input("Please input the id you want to use: ")
    if env_id == '1':
        single_env_test(device)
    elif env_id == '2':
        multi_env_test(device)
    else:
        raise ValueError("Invalid environment id.")
