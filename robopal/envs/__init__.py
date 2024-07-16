from .base import MujocoEnv
from .robot import RobotEnv
from .manipulation_tasks.robot_manipulate import ManipulateEnv

# single agent environments
from .manipulation_tasks.demo_pick_place import PickAndPlaceEnv
from .manipulation_tasks.demo_triple_stack import TripleStackEnv
from .manipulation_tasks.demo_drawer import DrawerEnv
from .manipulation_tasks.demo_cabinet import LockedCabinetEnv
from .manipulation_tasks.demo_cube_drawer import DrawerCubeEnv

# double agent environments
from .bimanual_tasks.bimanual_manipulate import BimanualManipulate
from .bimanual_tasks.bimanual_pick_place import BimanualPickAndPlace
from .bimanual_tasks.bimanual_reach import BimanualReach
from .bimanual_tasks.bimanual_transport import BimanualTransport

ENVS = [
    PickAndPlaceEnv,
    TripleStackEnv,
    DrawerEnv,
    LockedCabinetEnv,
    DrawerCubeEnv,
    BimanualReach,
    BimanualPickAndPlace,
    BimanualTransport,
]
REGISTERED_ENVS = {env.name: env for env in ENVS}
