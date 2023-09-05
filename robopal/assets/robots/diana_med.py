from robopal.commons import RobotGenerator
from robopal.assets.robots.base import *
import os


class DianaMed(ArmBase):
    def __init__(self):
        self.robot = RobotGenerator(
            scene='default',
            # chassis='Omnidirect',
            manipulator='DianaMed',
            # gripper='robotiq_gripper',
            g2m_body=['0_link7']
        )
        super().__init__(
            name="diana_med",
            urdf_path=os.path.join(os.path.dirname(__file__), "../models/manipulators/DianaMed/DianaMed.urdf"),
            xml_path=self.robot.xml,
            gripper=None
        )
        self.single_arm = self.Arm(self, 'single')
        self.single_arm.joint_index = ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7']
        self.single_arm.actuator_index = ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7']
        self.single_arm.setArmInitPose(self.init_qpos)
        self.arm.append(self.single_arm)

        self.jnt_num = self.single_arm.jnt_num

        print("DianaMed init !")

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])
