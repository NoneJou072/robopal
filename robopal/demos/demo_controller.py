import argparse
import numpy as np

from robopal.robots.diana_med import DianaMed
from robopal.envs import RobotEnv, PosCtrlEnv


parser = argparse.ArgumentParser()
parser.add_argument('--ctrl', default='JNTIMP', type=str, help='JSC for task space controller or OSC for joint space controller')
args = parser.parse_args()

assert args.ctrl in ['JNTIMP', 'JNTNONE', 'JNTVEL', 'CARTIMP', 'CARTIK']

if args.ctrl in ['JNTIMP', 'JNTNONE', 'JNTVEL', 'CARTIMP']:
    env = RobotEnv(
        robot=DianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        controller=args.ctrl,
    )

    if args.ctrl == 'JNTIMP' or args.ctrl == 'JNTNONE':
        action = np.array([0.33116, -0.39768533, 0.66947228, 0.33116, -0.39768533, 0.66947228, 0])

    elif args.ctrl == 'JNTVEL':
        action = np.array([0.01, -0.01, 0.0, 0.0, 0.01, 0.01, 0])

    elif args.ctrl == 'CARTIMP':
        action = np.array([0.33, -0.39, 0.66, 1.0, 0.0, 0.0, 0])

elif args.ctrl == 'CARTIK':
    env = PosCtrlEnv(
        robot=DianaMed(),
        render_mode='human',
        control_freq=200,
        is_interpolate=False,
        is_pd=False
    )
    action = np.array([0.33, -0.39, 0.66])

env.reset()
for t in range(int(1e4)):
    env.step(action)
    env.render()
env.close()
