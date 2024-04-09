from robopal.envs.parallel_dual_arm import BimanualPettingStyleEnv
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = BimanualPettingStyleEnv()
    parallel_api_test(env, num_cycles=1_000_000)
