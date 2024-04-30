import numpy as np

from robopal.robots.diana_med import DianaAssemble
from robopal.envs import RobotEnv

env = RobotEnv(
    robot=DianaAssemble(),
    render_mode='human',
    control_freq=100,
    is_interpolate=False,
    controller="CARTIK",
)
env.reset()

action = np.array([0.525, 0.13, 0.16, 1, 0, 0, 0])
for t in range(int(5e2)):
    env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = 0.03
    env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = 0.03
    env.step(action)

action = np.array([0.525, 0.13, 0.12, 1, 0, 0, 0])
for t in range(int(5e2)):
    env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = 0.03
    env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = 0.03
    env.step(action)

action = np.array([0.525, 0.13, 0.12, 1, 0, 0, 0])
for t in range(int(5e2)):
    env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = -0.02
    env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = -0.02
    env.step(action)

action = np.array([0.525, 0.13, 0.25, 1, 0, 0, 0])
for t in range(int(5e2)):
    env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = -0.02
    env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = -0.02
    env.step(action)

action = np.array([0.525, -0.112, 0.25, 1, 0, 0, 0])
for t in range(int(1e3)):
    env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = -0.02
    env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = -0.02
    env.step(action)

action = np.array([0.525, -0.112, 0.2, 1, 0, 0, 0])
for t in range(int(1e4)):
    env.mj_data.actuator('0_gripper_l_finger_joint').ctrl[0] = -0.02
    env.mj_data.actuator('0_gripper_r_finger_joint').ctrl[0] = -0.02
    env.step(action)

env.close()
