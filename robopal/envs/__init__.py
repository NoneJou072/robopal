from .base import MujocoEnv
from .robot import RobotEnv

from .manipulation_tasks.demo_pick_place import PickAndPlaceEnv
from .manipulation_tasks.demo_multi_cubes import MultiCubesEnv
from .manipulation_tasks.demo_drawer import DrawerEnv
from .manipulation_tasks.demo_cabinet import LockedCabinetEnv
from .manipulation_tasks.demo_cube_drawer import DrawerCubeEnv

ENVS = [
    PickAndPlaceEnv,
    MultiCubesEnv,
    DrawerEnv,
    LockedCabinetEnv,
    DrawerCubeEnv,
]
REGISTERED_ENVS = {env.name: env for env in ENVS}
