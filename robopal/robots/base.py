import abc

import mujoco
import numpy as np
from robopal.commons import RobotGenerator


class BaseArm:
    """ Base class for generating Data struct of the arm.

    :param name(str): robot name
    :param scene(str): scene name
    :param chassis(str): chassis name
    :param manipulator(str): manipulator name
    :param gripper(str): gripper name
    :param g2m_body(str): gripper to manipulator body name
    :param urdf_path(str): urdf file path
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

        self.joint_index = []
        self.actuator_index = []

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
    def add_assets(self) -> None:
        """ Add objects into the xml file. """
        pass

    @property
    def jnt_num(self) -> int:
        return len(self.joint_index)

    @property
    def arm_qpos(self) -> np.ndarray:
        return np.array([self.robot_data.joint(self.joint_index[i]).qpos[0] for i in range(self.jnt_num)])

    @property
    def arm_qvel(self) -> np.ndarray:
        return np.array([self.robot_data.joint(self.joint_index[i]).qvel[0] for i in range(self.jnt_num)])
