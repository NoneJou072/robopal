
# Wrapper: Stylize your environment

robopal 的环境接口具备 OpenAI [Gymnasium](https://gymnasium.farama.org/index.html) 的 API 风格。
通过结合 robopal 中提供的不同类型的 `wrapper`，可以方便地将仿真环境包装成 gym 环境，从而结合其他 RL 算法库进行训练。

## GymWrapper
pass
## GoalEnvWrapper

下面的示例演示了使用 [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) 训练 *Pick-and-Place* 任务。
其中，使用的环境来自于 `robopal/envs/manipulation_tasks/single_task_manipulation` 目录下的 `PickAndPlaceEnv` 环境。

<details>
  <summary><b>GoalEnvWrapper Showcase <span style="color:red;">(click to expand)</span></b></summary>
<p>

```python
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from robopal.demos.single_task_manipulation import PickAndPlaceEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 51200 == 0:
            self.model.save(self.log_dir + f"/model_saved/TQC/diana_pick_place_v2_{self.n_calls}")
        return True


log_dir = "log/"

env = PickAndPlaceEnv(
    render_mode="human",
    control_freq=10,
    controller='JNTIMP',
)
env = GoalEnvWrapper(env)

# Initialize the model
model = TQC(
    'MultiInputPolicy',
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log=log_dir,
    batch_size=1024,
    gamma=0.95,
    tau=0.05,
    policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
)

# Train the model
model.learn(int(1e6), callback=TensorboardCallback(log_dir=log_dir))

model.save("./her_bit_env")
```
</p>
</details>

## PettingStyleWrapper
下面的示例演示了使用 [RLlib]() 训练 *BimanualReach* 任务。
其中，使用的环境来自于 `robopal/envs/bimanual_tasks/bimanual_reach.py` 目录下的 `BimanualReach` 环境。
<details>
  <summary><b>PettingStyleWrapper Showcase <span style="color:red;">(click to expand)</span></b></summary>
<p>

```python
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

```
</p>
</details>