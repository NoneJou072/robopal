import abc

import mujoco
import numpy as np
from robopal.commons import RobotGenerator


class BaseArm:
    """ Base class for generating Data struct of the arm.

    :param name(str): robot name
    :param scene(str):
    :param chassis(str):
    :param manipulator(str):
    :param gripper(str):
    :param g2m_body(str):
    :param urdf_path(str):
    :param xml_path(str): If you have specified the xml path of your local robot,
    it'll not automatically construct the xml file with input assets.
    """

    def __init__(self,
                 name: str = None,
                 scene: str = 'default',
                 chassis: str = None,
                 manipulator: str = None,
                 gripper: str = None,
                 g2m_body: list = None,
                 urdf_path: str = None,
                 ):
        self.name = name
        self.scene = scene
        self.chassis = chassis
        self.manipulator = manipulator
        self.gripper = gripper
        self.g2m_body = g2m_body

        self.urdf_path = urdf_path  # urdf file used for pinocchio lib

        self.mjcf_generator: RobotGenerator
        self.robot_model = None
        self.robot_data = None
        self.construct_mjcf_data()

        self.arm = []

    def construct_mjcf_data(self):
        self.mjcf_generator = RobotGenerator(
            scene=self.scene,
            chassis=self.chassis,
            manipulator=self.manipulator,
            gripper=self.gripper,
            g2m_body=self.g2m_body
        )
        self.add_assets()
        xml_path = self.mjcf_generator.save_and_load_xml()
        self.robot_model = mujoco.MjModel.from_xml_path(filename=xml_path, assets=None)
        self.robot_data = mujoco.MjData(self.robot_model)

    @abc.abstractmethod
    def add_assets(self):
        """ Add objects into the xml file. """
        pass

    class PartArm:
        """
        internal class, using for precise arm state, double (left and right) or single
        """

        def __init__(self, base, state):
            self.robot_model = base.robot_model
            self.robot_data = base.robot_data
            self.state = state

            self.joint_index = []
            self.actuator_index = []
            self.base_joint_index = []
            self.base_actuator_index = []

            self.init_pose = None
            self.init_base_pose = None

        @property
        def jnt_num(self):
            return len(self.joint_index)

        def getBaseJoint(self):
            """base joints are the rotation joints in robot stand """
            self.base_joint_index = [string for string in self.joint_index if 'base' in string]
            return self.base_joint_index

        def getBaseActuator(self):
            """base actuators are the rotation motors in robot stand """
            self.base_actuator_index = [string for string in self.actuator_index if 'base' in string]
            return self.base_actuator_index

        def setArmInitPose(self, initPose: np.array):
            if len(initPose) == self.jnt_num:
                self.init_pose = initPose
            else:
                raise ValueError("shape error for input pose vector")  # need to optimize

        def setBaseInitPose(self, initBasePose: np.array):
            if len(self.init_base_pose) == len(initBasePose):
                self.init_base_pose = initBasePose
            else:
                raise ValueError("shape error for input pose vector")  # need to optimize

        def get_Arm_index(self):
            joint_index = []
            for i in range(self.jnt_num):
                joint_index.append(self.joint_index[i])
            return joint_index

        def get_Arm_id(self):
            joint_id = []
            for i in range(self.jnt_num):
                joint_id.append(mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_index[i]))
            return joint_id

        @property
        def arm_qpos(self):
            qpos_states = []
            for i in range(self.jnt_num):
                qpos_states.append(self.robot_data.joint(self.joint_index[i]).qpos[0])
            return np.array(qpos_states)

        @arm_qpos.setter
        def arm_qpos(self, value):
            raise ValueError("You cannot set qpos directly.")

        @property
        def arm_qvel(self):
            qvel_states = []
            for i in range(self.jnt_num):
                qvel_states.append(self.robot_data.joint(self.joint_index[i]).qvel[0])
            return np.array(qvel_states)

        @property
        def arm_torq(self):
            qtorq_states = []
            for i in range(self.jnt_num):
                qtorq_states.append(self.robot_data.actuator(self.actuator_index[i]).ctrl[0])
            return qtorq_states
