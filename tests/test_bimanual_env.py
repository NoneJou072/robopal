import argparse
import numpy as np
import logging

from robopal.envs import RobotEnv

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='CARTIK', type=str,
                    help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

assert args.ctrl in ['JNTIMP', 'CARTIK'], 'Invalid controller'

if args.ctrl == 'JNTIMP':
    actions = [np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0]),
               np.array([0.3, -0.4, 0.7, 0.3, -0.4, 0.7, 0])]

elif args.ctrl == 'CARTIK':
    actions = [np.array([0.3, 0.4, 0.4, 1, 0, 0, 0]),
               np.array([0.4, -0.4, 0.5, 1, 0, 0, 0])]
else:
    raise ValueError('Invaild controller.')

env = RobotEnv(
    "DualPanda",
    render_mode='human',
    control_freq=200,
    controller=args.ctrl,
)
actions = {agent: actions[id] for id, agent in enumerate(env.agents)}

env.reset()
for t in range(int(1e4)):
    env.step(actions)
env.close()
