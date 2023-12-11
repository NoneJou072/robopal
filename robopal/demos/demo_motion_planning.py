import numpy as np
from robopal.robots.diana_med import DianaCollide
from robopal.envs import PosCtrlEnv
from robopal.controllers.rrt import rrt_star


env = PosCtrlEnv(
    robot=DianaCollide(),
    renderer='viewer',
    is_render=True,
    control_freq=200,
    is_interpolate=False,
    is_pd=False
)
env.reset()

current_pos, _ = env.kdl_solver.fk(env.robot.arm_qpos)
env.save_state()
path = rrt_star(current_pos, [0.6, 0.5, 0.4], env)
env.load_state()
assert path is not None

for pos in reversed(path):
    pos = np.array(pos)
    print(pos)
    current_pos, _ = env.kdl_solver.fk(env.robot.arm_qpos)
    while np.linalg.norm(current_pos - pos) > 0.02:
        env.step(np.concatenate([pos, [1, 0, 0, 0]]))
        current_pos, _ = env.kdl_solver.fk(env.robot.arm_qpos)
        env.render()
        env.renderer.add_visual_point(path)

print("finished.")
for t in range(int(1e4)):
    env.render()

env.close()
