from .diana_med import DianaMed
from robopal.robots.grippers import *

END_MAP = {
    "rethink_gripper": RethinkGripper,
    "robotiq_gripper": RobotiqGripper,
    "panda_hand": PandaHand,
}
