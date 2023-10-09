from robopal.demos.demo_pick_place import PickAndPlaceEnv
from robopal.demos.demo_drawer import DrawerEnv
from robopal.demos.demo_cube_drawer import DrawerCubeEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper

from gymnasium.envs.registration import register

register(
    id='PickAndPlace-v1',
    entry_point='robopal.demos:GoalEnvWrapper',
    max_episode_steps=50,
    kwargs={'env': PickAndPlaceEnv()}
)

register(
    id='Drawer-v1',
    entry_point='robopal.demos:DrawerEnv',
    max_episode_steps=50
)

register(
    id='DrawerCube-v1',
    entry_point='robopal.demos:DrawerCubeEnv',
    max_episode_steps=50
)
