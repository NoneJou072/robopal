import os

from robopal.robots.base import *

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')


class Panda(BaseRobot):
    """ DianaMed robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='Panda',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            name="diana_med",
            scene=scene,
            chassis=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body='0_attachment',
        )
        self.arm_joint_names = {self.agents[0]: ['0_joint1', '0_joint2', '0_joint3', '0_joint4', '0_joint5', '0_joint6', '0_joint7']}
        self.arm_actuator_names = {self.agents[0]: ['0_actuator1', '0_actuator2', '0_actuator3', '0_actuator4', '0_actuator5', '0_actuator6', '0_actuator7']}
        self.base_link_name = {self.agents[0]: '0_link0'}
        self.end_name = {self.agents[0]: '0_attachment'}

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])}
