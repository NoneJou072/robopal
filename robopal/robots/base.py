import abc
import logging

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
                 chassis: str | list[str] = None,
                 manipulator: str | list[str] = None,
                 gripper: str | list[str] = None,
                 g2m_body: str | list[str] = None,
                 urdf_path: str = None,
                 ):
        self.name = name

        manipulator = [manipulator] if isinstance(manipulator, str) else manipulator
        self.agent_num = len(manipulator)
        self.agents = [f'arm{i}' for i in range(self.agent_num)]
        logging.info(f'Activated agents: {self.agents}')

        self._scene = scene
        self._chassis = chassis
        self._manipulator = manipulator
        self._gripper = gripper
        self._g2m_body = g2m_body

        self.urdf_path = urdf_path  # urdf file used for pinocchio lib

        self.mjcf_generator = self._construct_mjcf_data()
        self.add_assets()

        xml_path = self.mjcf_generator.save_and_load_xml()
        self.robot_model = mujoco.MjModel.from_xml_path(filename=xml_path, assets=None)
        self.robot_data = mujoco.MjData(self.robot_model)

        self.joint_index = [[]]
        self.actuator_index = [[]]

    def _construct_mjcf_data(self) -> RobotGenerator:
        return RobotGenerator(
            scene=self._scene,
            chassis=self._chassis,
            manipulator=self._manipulator,
            gripper=self._gripper,
            g2m_body=self._g2m_body
        )

    @property
    def init_qpos(self) -> np.ndarray:
        """ Robot's init joint position. """
        raise NotImplementedError

    @abc.abstractmethod
    def add_assets(self) -> None:
        """ Add objects into the xml file. """
        pass

    @property
    def jnt_num(self) -> int | dict[str, int]:
        """ Number of joints. """
        return len(self.joint_index[0])

    def get_arm_qpos(self, agent_index: int = 0) -> np.ndarray:
        """ Get arm joint position of the specified agent.

        :param agent_index: agent name
        :return: joint position
        """
        return np.array([self.robot_data.joint(j).qpos[0] for j in self.joint_index[agent_index]])

    def get_arm_qvel(self, agent_index: int = 0) -> np.ndarray:
        """ Get arm joint velocity of the specified agent.

        :param agent_index: agent name
        :return: joint position
        """
        return np.array([self.robot_data.joint(j).qvel[0] for j in self.joint_index[agent_index]])

    def get_arm_qacc(self, agent_index: int = 0) -> np.ndarray:
        """ Get arm joint accelerate of the specified agent.

        :param agent_index: agent name
        :return: joint position
        """
        return np.array([self.robot_data.joint(j).qacc[0] for j in self.joint_index[agent_index]])
