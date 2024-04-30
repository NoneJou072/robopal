import argparse
import numpy as np
import logging

from robopal.robots.diana_med import DianaMed
from robopal.envs import RobotEnv, PosCtrlEnv

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    options = {}

    # Choose controller
    options['ctrl'] = 'CARTIMP'

    assert options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'], 'Invalid controller'

    if options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP']:
        env = RobotEnv(
            robot=DianaMed(),
            render_mode='human',
            control_freq=200,
            is_interpolate=False,
            controller=options['ctrl'],
        )

        if options['ctrl'] == 'JNTIMP':
            action = np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])

        elif options['ctrl'] == 'JNTVEL':
            action = np.array([0.00, 0.1, 0.0, 0.0, 0., 0., 0])

        elif options['ctrl'] == 'CARTIMP':
            action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

    elif options['ctrl'] == 'CARTIK':
        env = PosCtrlEnv(
            robot=DianaMed(),
            render_mode='human',
            control_freq=200,
            is_interpolate=False,
            is_pd=False
        )
        action = np.array([0.33, -0.39, 0.66, 1, 0, 0, 0])
    else:
        raise ValueError('Invalid controller')

    if isinstance(env, RobotEnv):
        env.reset()
        for t in range(int(2e4)):
            env.step(action)
        env.close()
