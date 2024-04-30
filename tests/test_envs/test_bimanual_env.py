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

if args.ctrl not in ['JNTIMP', 'CARTIK']:
    raise ValueError('Invalid controller')

if args.ctrl == 'JNTIMP':
    env = RobotEnv(
        robot=DualDianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=args.ctrl,
    )

    actions = [np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0]),
               np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])]

elif args.ctrl == 'CARTIK':
    env = PosCtrlEnv(
        robot=DualDianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        is_pd=True
    )
    actions = [np.array([0.3, 0.4, 0.4, 1, 0, 0, 0]),
               np.array([0.4, -0.4, 0.6, 1, 0, 0, 0])]
else:
    raise ValueError('Invaild controller.')

actions = {agent: actions[id] for id, agent in enumerate(env.agents)}

if isinstance(env, RobotEnv):
    env.reset()
    for t in range(int(1e4)):
        env.step(actions)
    env.close()
