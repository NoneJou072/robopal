from .base import MujocoEnv
from .robot import RobotEnv
from .manipulation_tasks.robot_manipulate import ManipulateEnv

# single agent environments
from .manipulation_tasks.demo_pick_place import PickAndPlaceEnv
from .manipulation_tasks.demo_multi_cubes import MultiCubesEnv
from .manipulation_tasks.demo_drawer import DrawerEnv
from .manipulation_tasks.demo_cabinet import LockedCabinetEnv
from .manipulation_tasks.demo_cube_drawer import DrawerCubeEnv

# double agent environments
from .bimanual_tasks.bimanual_manipulate import BimanualManipulate
from .bimanual_tasks.bimanual_pick_place import BimanualPickAndPlace
from .bimanual_tasks.bimanual_reach import BimanualReach

ENVS = [
    PickAndPlaceEnv,
    MultiCubesEnv,
    DrawerEnv,
    LockedCabinetEnv,
    DrawerCubeEnv,
    BimanualReach,
    BimanualPickAndPlace,
]
REGISTERED_ENVS = {env.name: env for env in ENVS}
