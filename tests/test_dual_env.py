import argparse
import numpy as np
import logging

from robopal.robots.diana_med import DualDianaMed
from robopal.envs import RobotEnv, PosCtrlEnv

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='CARTIK', type=str,
                    help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

if args.ctrl not in ['JNTIMP', 'JNTNONE', 'JNTVEL', 'CARTIMP', 'CARTIK']:
    raise ValueError('Invalid controller')

if args.ctrl in ['JNTIMP', 'JNTNONE', 'JNTVEL', 'CARTIMP']:
    env = RobotEnv(
        robot=DualDianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=args.ctrl,
    )

    if args.ctrl == 'JNTIMP':
        action = np.array([0.33, -0.4, 0.67, 0.33, -0.4, 0.67, 0])

    elif args.ctrl == 'JNTVEL':
        action = np.array([0.01, -0.01, 0.0, 0.0, 0.01, 0.01, 0])

    else:  # args.ctrl == 'CARTIMP'
        action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

else:  # args.ctrl == 'CARTIK'
    env = PosCtrlEnv(
        robot=DualDianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        is_pd=False
    )
    action = np.array([0.33, -0.39, 0.66, 1, 0, 0, 0])

actions = {agent: action for agent in env.agents}

if isinstance(env, RobotEnv):
    env.reset()
    for t in range(int(1e4)):
        env.step(actions)
        env.render()
    env.close()
