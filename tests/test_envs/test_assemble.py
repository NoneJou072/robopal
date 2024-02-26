import argparse
import numpy as np
import logging

from robopal.robots.diana_med import DianaAssemble
from robopal.envs import RobotEnv, PosCtrlEnv

env = PosCtrlEnv(
    robot=DianaAssemble(),
    render_mode='human',
    control_freq=200,
    is_interpolate=False,
    is_pd=False
)
action = np.array([0.33, -0.39, 0.66, 1, 0, 0, 0])

env.reset()
for t in range(int(1e6)):
    env.step(action)
env.close()
