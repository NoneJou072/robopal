import os

from robopal.envs.bimanual_tasks import BimanualReach
from robopal.wrappers import PettingStyleWrapper
import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from pettingzoo.utils import parallel_to_aec


def env_creator():
    env = BimanualReach(render_mode='human')
    env = PettingStyleWrapper(env)
    env = parallel_to_aec(env)
    return env


if __name__ == "__main__":


    env = env_creator()
    env_name = "bimanualreach_v0"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))

    ray.init()

    checkpoint_path = "<your checkpoint dir path>"
    PPOagent = PPO.from_checkpoint(checkpoint_path)

    reward_sum = 0

    for _ in range(1000):
        env.reset()

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            reward_sum += reward
            if termination or truncation:
                action = None
            else:
                action = PPOagent.compute_single_action(observation)
            print(agent, action)

            env.step(action)

    env.close()

    print(reward_sum)
