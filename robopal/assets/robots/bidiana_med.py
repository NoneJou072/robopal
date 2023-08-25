from robopal.assets.robots.arm_base import *
import os
from robopal.utils import RobotGenerator


class BiDianaMed(ArmBase):
    def __init__(self):
        robot = RobotGenerator(
            name='robot',
            scene='default',
            # chassis='Omnidirect',
            manipulator='Bimanual',
            gripper='robotiq_gripper',
            g2m_body=['0_right_link7', '0_left_link7']
        )
        super().__init__(
            name="bidiana_med",
            urdf_path=os.path.join(os.path.dirname(__file__), "../models/manipulators/DianaMed/DianaMed.urdf"),
            xml_path=robot.xml,
            gripper=None
        )
        self.left_arm = self.Arm(self, 'left')
        self.right_arm = self.Arm(self, 'right')
        self.arm_base = self.Arm(self, 'single')
        self.left_arm.joint_index = ['0_left_j1', '0_left_j2', '0_left_j3', '0_left_j4', '0_left_j5', '0_left_j6', '0_left_j7']
        self.left_arm.actuator_index = ['0_left_m1', '0_left_m2', '0_left_m3', '0_left_m4', '0_left_m5', '0_left_m6', '0_left_m7']
        self.right_arm.joint_index = ['0_right_j1', '0_right_j2', '0_right_j3', '0_right_j4', '0_right_j5', '0_right_j6', '0_right_j7']
        self.right_arm.actuator_index = ['0_right_m1', '0_right_m2', '0_right_m3', '0_right_m4', '0_right_m5', '0_right_m6', '0_right_m7']
        self.base_joint_index = ['0_base_stand_joint']
        self.base_actuator_index = ['0_base_stand_joint_motor']
        self.left_arm.setArmInitPose(self.init_left_qpos)
        self.right_arm.setArmInitPose(self.init_right_qpos)
        self.arm.extend((self.left_arm, self.right_arm))
        self.arm_state = len(self.arm)  # default order: left first, then right !
        self.jnt_num = self.left_arm.jnt_num + self.right_arm.jnt_num + self.arm_base.jnt_num

        # joint PD
        # self.kp = np.array([2, 2, 2, 2, 2, 2, 2])
        # self.kp = np.array([1000, 800, 600, 600, 600, 600, 600])
        self.kp = 625 * np.ones(self.jnt_num)
        self.kd = 275 * np.ones(self.jnt_num)
        print("BiDianaMed init !")

    @property
    def init_left_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, -np.pi / 1.5, 0.0, np.pi / 4.0, 0.00, np.pi / 4.0, 0.0])
    
    @property
    def init_right_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, -np.pi / 1.5, 0.0, np.pi / 4.0, 0.00, np.pi / 4.0, 0.0])
