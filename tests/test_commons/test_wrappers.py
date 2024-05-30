from robopal.envs.manipulation_tasks.demo_pick_place import PickAndPlaceEnv
from robopal.wrappers.gym_wrapper import GoalEnvWrapper

env = PickAndPlaceEnv(render_mode="human")
env = GoalEnvWrapper(env)
env.reset()
for t in range(int(1e4)):
    action = env.action_space.sample()
    env.step(action)
env.close()
