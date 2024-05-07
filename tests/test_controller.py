import argparse
import numpy as np
import logging

from robopal.robots.diana_med import DianaMed
from robopal.robots.panda import Panda
from robopal.envs import RobotEnv

logging.basicConfig(level=logging.INFO)

    
if __name__ == "__main__":

    options = {}

    # Choose controller
    options['ctrl'] = 'CARTIK'

    assert options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'], 'Invalid controller'

    env = RobotEnv(
        robot=DianaMed,
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=options['ctrl'],
    )

    if options['ctrl'] == 'JNTIMP':
        action = np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])

    elif options['ctrl'] == 'JNTVEL':
        action = np.array([0.1, 0.1, 0.0, 0.0, 0., 0., 0])

    elif options['ctrl'] == 'CARTIMP':
        action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

    elif options['ctrl'] == 'CARTIK':
        action = np.array([0.33, -0.3, 0.5, 1, 0, 0, 0])

    def test_JNTIMP_error():
        print(np.linalg.norm(action - env.robot.get_arm_qpos()))
    
    def test_CARTIK_error():
        current_pos, current_quat = env.controller.forward_kinematics(env.robot.get_arm_qpos())
        print(np.sum(np.abs(action[:3] - current_pos)) + np.sum(np.abs(action[3:] - current_quat)))

    if isinstance(env, RobotEnv):
        env.reset()
        for t in range(int(2e4)):
            env.step(action)
            test_CARTIK_error()
        env.close()
