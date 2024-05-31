import numpy as np

from robopal.robots.diana_med import DianaCollide
from robopal.envs import RobotEnv
from robopal.controllers.planners.rrt import rrt_star


if __name__ == "__main__":
    env = RobotEnv(
        robot=DianaCollide,
        render_mode="human",
        control_freq=200,
        is_interpolate=False,
        controller="CARTIK"
    )
    env.reset()

    goal_pos = [0.6, 0.5, 0.4]
    current_pos = env.robot.get_end_xpos()

    path = rrt_star(current_pos, goal_pos, env)
    assert path is not None

    for pos in reversed(path):
        pos = np.array(pos)
        current_pos = env.robot.get_end_xpos()
        while np.linalg.norm(current_pos - pos) > 0.02:
            env.renderer.add_visual_point(path)

            env.step(np.concatenate([pos, [1, 0, 0, 0]]))
            current_pos = env.robot.get_end_xpos()

    print("finished.")

    for t in range(int(1e4)):
        env.render()

    env.close()
