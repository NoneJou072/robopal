import numpy as np
from robopal.envs.jnt_ctrl_env import SingleArmEnv

if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = SingleArmEnv(
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
