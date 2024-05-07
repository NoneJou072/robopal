import os

from robopal.robots.base import *

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')


class UR5e(BaseRobot):
    """ UR5e robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='UR5e',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            name="ur5e",
            scene=scene,
            chassis=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body='0_attachment',
        )
        self.arm_joint_names = {self.agents[0]: ['0_shoulder_pan_joint', '0_shoulder_lift_joint', '0_elbow_joint', '0_wrist_1_joint', '0_wrist_2_joint', '0_wrist_3_joint']}
        self.arm_actuator_names = {self.agents[0]: ['0_actuator1', '0_actuator2', '0_actuator3', '0_actuator4', '0_actuator5', '0_actuator6']}
        self.base_link_name = {self.agents[0]: '0_base'}
        self.end_name = {self.agents[0]: '0_attachment'}

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0])}
