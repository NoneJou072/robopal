from stable_baselines3 import HerReplayBuffer, SAC
# from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from robopal.envs.manipulation_tasks.demo_pick_place import PickAndPlaceEnv
from robopal.wrappers import GoalEnvWrapper


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 51200 == 0:
            self.model.save(self.log_dir + f"/model_saved/SAC/diana_pick_place_v2_{self.n_calls}")
        return True


log_dir = "log/"

env = PickAndPlaceEnv(render_mode=None)
env = GoalEnvWrapper(env)

# Initialize the model
model = SAC(
    'MultiInputPolicy',
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        copy_info_dict=True,
    ),
    verbose=1,
    tensorboard_log=log_dir,
    batch_size=1024,
    learning_rate=1e-3,
    gamma=0.95,
    tau=0.05,
    policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
)

# Train the model
model.learn(int(1e6), callback=TensorboardCallback(log_dir=log_dir))

model.save("./her_bit_env")
