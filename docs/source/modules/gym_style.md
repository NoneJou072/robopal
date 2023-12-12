
# Gymnasium Style Interface

我们的环境遵循 OpenAI Gymnasium 接口规范，可以方便地搭建强化学习训练环境。

下面是一个简单的示例，演示了使用 [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) 训练 *Pick-and-Place* 任务。
其中，使用的环境来自于 `robopal/demos/single_task_manipulation` 目录下的 `PickAndPlaceEnv` 环境。

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
下·