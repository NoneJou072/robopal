import numpy as np

from robopal.robots.diana_med import DianaAssemble
from robopal.envs import RobotEnv

env = RobotEnv(
    robot=DianaAssemble,
    render_mode='human',
    control_freq=100,
    is_interpolate=False,
    controller="CARTIK",
)
env.reset()

action = np.array([0.525, 0.13, 0.16, 1, 0, 0, 0])
for t in range(200):
    env.robot.end['arm0'].open()
    env.step(action)

action = np.array([0.525, 0.13, 0.12, 1, 0, 0, 0])
for t in range(200):
    env.robot.end['arm0'].open()
    env.step(action)

action = np.array([0.525, 0.13, 0.12, 1, 0, 0, 0])
for t in range(200):
    env.robot.end['arm0'].close()
    env.step(action)

action = np.array([0.525, 0.13, 0.25, 1, 0, 0, 0])
for t in range(200):
    env.robot.end['arm0'].close()
    env.step(action)

action = np.array([0.525, -0.112, 0.25, 1, 0, 0, 0])
for t in range(200):
    env.robot.end['arm0'].close()
    env.step(action)

action = np.array([0.525, -0.112, 0.2, 1, 0, 0, 0])
for t in range(200):
    env.robot.end['arm0'].close()
    env.step(action)
action = np.array([0.525, -0.112, 0.2, 1, 0, 0, 0])
for t in range(int(1e4)):
    env.robot.end['arm0'].open()
    env.step(action)

env.close()
