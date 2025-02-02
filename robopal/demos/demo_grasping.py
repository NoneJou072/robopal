import numpy as np

from robopal.envs import RobotEnv
from robopal.robots.ur5e import UR5eGrasp
from robopal.robots.diana_med import DianaGrasp
from robopal.robots.panda import PandaGrasp

GRASIPING_ROBOTS = [UR5eGrasp, DianaGrasp, PandaGrasp]

if __name__ == "__main__":
    for robot in GRASIPING_ROBOTS:
        env = RobotEnv(
            robot,
            render_mode='human',
            control_freq=100,
            controller='CARTIK',
        )
        env.reset()
        env.controller.reference = 'world'

        env.robot.end['agent0'].open()
        action = env.get_body_pos('green_block')
        for t in range(int(200)):
            env.step(action)
        for t in range(int(250)):
            env.robot.end['agent0'].close()
            env.step(action)
        action += np.array([0.0, 0.0, 0.29])
        for t in range(int(200)):
            env.step(action)
        env.renderer.close_render_window()
    env.close()
