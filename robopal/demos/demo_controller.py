import numpy as np
import logging

from robopal.robots.diana_med import DianaMed
from robopal.robots.dual_arms import DualDianaMed
from robopal.robots.panda import Panda
from robopal.robots.ur5e import UR5e
from robopal.envs import RobotEnv

logging.basicConfig(level=logging.INFO)

MANIPULATORS = {
    'DianaMed': DianaMed,
    'Panda' : Panda,
    'UR5e' : UR5e,
    'DualDianaMed': DualDianaMed,
}

if __name__ == "__main__":

    options = {}
    
    # Choose the manipulator
    print("Available manipulators:\n 1. DianaMed\n 2. Panda\n 3. UR5e\n 4. DualDianaMed")
    manipulator_id = input("Choose the manipulator:")
    options['manipulator'] = list(MANIPULATORS.keys())[int(manipulator_id) - 1]
    assert options['manipulator'] in MANIPULATORS.keys(), 'Invalid manipulator'
    options['manipulator'] = MANIPULATORS[options['manipulator']]

    # Choose the controller
    if options['manipulator'] == DualDianaMed:
        print("Available controllers:\n 1.JNTIMP\n 2.CARTIK")
        ctrl_id = input("Choose the controller:")
        options['ctrl'] = ['JNTIMP', 'CARTIK'][int(ctrl_id) - 1]
        assert options['ctrl'] in ['JNTIMP', 'CARTIK'], 'Invalid controller'
    else:
        print("Available controllers:\n 1.JNTTAU\n 2.JNTIMP\n 3.JNTVEL\n 4.CARTIMP\n 5.CARTIK")
        ctrl_id = input("Choose the controller:")
        options['ctrl'] = ['JNTTAU', 'JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'][int(ctrl_id) - 1]
        assert options['ctrl'] in ['JNTTAU', 'JNTIMP', 'JNTVEL', 'CARTIMP', 'CARTIK'], 'Invalid controller'

    # Initialize the environment
    env = RobotEnv(
        robot=options['manipulator'],
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=options['ctrl'],
    )

    # Set the action
    if env.robot.agent_num == 1:
        if options['ctrl'] == 'JNTTAU':
            action = 10 * np.ones(env.robot.jnt_num)
            
        elif options['ctrl'] == 'JNTIMP':
            action = np.random.uniform(low=env.robot.mani_joint_bounds[env.robot.agents[0]][0],
                                       high=env.robot.mani_joint_bounds[env.robot.agents[0]][1],
                                       size=env.robot.jnt_num)

        elif options['ctrl'] == 'JNTVEL':
            action = 0.1 * np.ones(env.robot.jnt_num)

        elif options['ctrl'] == 'CARTIMP':
            action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

        elif options['ctrl'] == 'CARTIK':
            action = np.array([0.33, -0.3, 0.5, 1, 0, 0, 0])
    else:
        if options['ctrl'] == 'JNTIMP':
            action = [np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0]),
                    np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])]

        elif options['ctrl'] == 'CARTIK':
            action = [np.array([0.3, 0.4, 0.4, 1, 0, 0, 0]),
                    np.array([0.4, -0.4, 0.6, 1, 0, 0, 0])]
            
        action = {agent: action[id] for id, agent in enumerate(env.agents)}
    
    # Main loop
    env.reset()
    for _ in range(int(2e4)):
        env.step(action)
    env.close()
