import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctrl', type=str, help='JSC for task space controller or OSC for joint space controller')
    args = parser.parse_args()

    if args.env == 'JSC':
        from robopal.assets.robots.diana_med import DianaMed
        from robopal.envs import JntCtrlEnv

        env = JntCtrlEnv(
            robot=DianaMed(),
            renderer='viewer',
            is_render=True,
            control_freq=200,
            is_interpolate=True,
        )
        env.reset()
        for t in range(int(1e6)):
            action = np.array([0.33116, -0.39768533, 0.66947228, 0.33116, -0.39768533, 0.66947228, 0])
            env.step(action)
            if env.is_render:
                env.render()

    elif args.env == 'OSC':
        from robopal.assets.robots.diana_med import DianaMed
        from robopal.envs import PosCtrlEnv

        env = PosCtrlEnv(
            robot=DianaMed(),
            renderer='viewer',
            is_render=True,
            control_freq=200,
            is_interpolate=True,
            is_pd=True
        )
        env.reset()
        for t in range(int(1e6)):
            action = np.array([0.33116, -0.39768533, 0.66947228])
            env.step(action)
            if env.is_render:
                env.render()
