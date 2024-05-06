import numpy as np
import logging

from robopal.robots.diana_med import DianaMed
from robopal.robots.panda import Panda
from robopal.envs import RobotEnv

logging.basicConfig(level=logging.INFO)

MANIPULATORS = {
    'DianaMed': DianaMed,
    'Panda' : Panda,
}

if __name__ == "__main__":

    options = {}
    # Choose the manipulator
    print("available manipulator:\n DianaMed, Panda")
    options['manipulator'] = input("Choose the manipulator:")
    assert options['manipulator'] in MANIPULATORS.keys(), 'Invalid manipulator'
    options['manipulator'] = MANIPULATORS[options['manipulator']]

    # Choose the controller
    print("available controllers:\n JNTIMP, JNTVEL, CARTIMP, CARTIK")
    options['ctrl'] = input("Choose the controller:")
    assert options['ctrl'] in ['JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'], 'Invalid controller'

    env = RobotEnv(
        robot=options['manipulator'],
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
        action = np.array([0.33, -0.3, 0.5, 1, 0, 0, 0])

    env.reset()
    for _ in range(int(2e4)):
        env.step(action)
    env.close()
