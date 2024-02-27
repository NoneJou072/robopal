import numpy as np

from robopal.robots.diana_med import DianaCollide
from robopal.envs import PosCtrlEnv
from robopal.controllers.rrt import rrt_star


env = PosCtrlEnv(
    robot=DianaCollide(),
    render_mode="human",
    control_freq=200,
    is_interpolate=False,
    is_pd=False
)
env.reset()

goal_pos = [0.6, 0.5, 0.4]
current_pos, _ = env.kd_solver.fk(env.robot.get_arm_qpos())

path = rrt_star(current_pos, goal_pos, env)
assert path is not None

for pos in reversed(path):
    pos = np.array(pos)
    print(pos)
    current_pos, _ = env.kd_solver.fk(env.robot.get_arm_qpos())
    while np.linalg.norm(current_pos - pos) > 0.02:
        env.renderer.set_renderer_config()
        env.renderer.add_visual_point(path)

        env.step(np.concatenate([pos, [1, 0, 0, 0]]))
        current_pos, _ = env.kd_solver.fk(env.robot.get_arm_qpos())

print("finished.")

for t in range(int(1e4)):
    env.render()

env.close()
