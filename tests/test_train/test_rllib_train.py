import os

from robopal.envs.bimanual_tasks import BimanualReach
from robopal.wrappers import PettingStyleWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env


def env_creator(args):
    env = BimanualReach(render_mode=None)
    env = PettingStyleWrapper(env)
    return env


if __name__ == "__main__":

    ray.init()

    env_name = "bimanualreach_v0"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=1e-4,
            gamma=0.96,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
