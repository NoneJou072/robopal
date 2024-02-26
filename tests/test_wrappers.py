from robopal.demos.single_task_manipulation import PickAndPlaceEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper

env = PickAndPlaceEnv(render_mode="human")
env = GoalEnvWrapper(env)
env.reset()
for t in range(int(1e4)):
    action = env.action_space.sample()
    env.step(action)
env.close()
