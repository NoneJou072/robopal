from robopal.demos.bimanual_tasks import BimanualPickAndPlace
from robopal.commons.wrappers import PettingStyleWrapper
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = PettingStyleWrapper(BimanualPickAndPlace(render_mode='human'))
    parallel_api_test(env, num_cycles=env.max_episode_steps)
