import abc
import logging
from typing import Union, List, Dict, Iterable
import copy

import mujoco
import numpy as np

import robopal
from robopal.commons.xml_splice import RobotGenerator
import robopal.robots
from robopal.robots.grippers import BaseEnd

REGISTERED_ROBOTS = {}


class RobotMetaClass(type):
    """Metaclass for registering robot arms"""

    def __new__(meta, name, bases, attrs):
        cls = super().__new__(meta, name, bases, attrs)

        if not cls.__name__ == "BaseRobot":
            REGISTERED_ROBOTS[cls.__name__] = cls
        return cls


class BaseRobot(metaclass=RobotMetaClass):
    """ Base class for generating Data struct of the arm.

    :param name(str): robot name
    :param scene(str): scene name
    :param mount(str): mount name
    :param manipulator(str): manipulator name
    :param gripper(str): gripper name
    :param attached_body(str): Connect the end of manipulator and gripper at this body.
    :param xml_path(str): If you have specified the xml path of your local robot,
    it'll not automatically construct the xml file with input assets.
    """

    def __init__(self,
                 scene: str = 'default',
                 mount: Union[str, Iterable[str]] = None,
                 manipulator: Union[str, Iterable[str]] = None,
                 gripper: Union[str, Iterable[str]] = None,
                 attached_body: Union[str, Iterable[str]] = None,
                 specified_xml_path: str = None,
                 agent_num: int = 0
                 ):

        self.specified_xml_path = specified_xml_path

        self.mjcf_generator = RobotGenerator(
                scene=scene,
                mount=mount,
                manipulator=manipulator,
                gripper=gripper,
                attached_body=attached_body,
                xml_path=specified_xml_path,
        )

        self.agent_num = agent_num  # by default, user should specify the agent number if not using the xml file.

        if specified_xml_path is None:
            manipulator = [manipulator] if isinstance(manipulator, str) else manipulator

            self.add_assets()

            self.agent_num = len(manipulator)
            xml_path = self.mjcf_generator.save_xml()
        else:
            assert self.agent_num > 0, 'Please specify the agent number by setting `self.agent_num`.'
            xml_path = self.mjcf_generator.get_xml_path()

        self.robot_model: mujoco.MjModel = None
        self.robot_data: mujoco.MjData = None
        # deepcopy for computing kinematics.
        self.kine_data: mujoco.MjData = None

        self.agents = [f'agent{i}' for i in range(self.agent_num)]
        logging.info(f'Activated agents: {self.agents}')
        
        self.build_from_xml(xml_path)

        self.create_end_effector(gripper)

        # manipulator infos
        self._arm_joint_names = dict()
        self._arm_joint_indexes = dict()
        self._arm_actuator_names = dict()
        self._arm_actuator_indexes = dict()
        self.base_link_name = dict()
        self.end_name = dict()
        self.mani_joint_bounds = dict()  # Bounds at the joint limits
        # Bounds at the workspace limits
        self.pos_max_bound = np.ones(3)
        self.pos_min_bound = -1.0 * np.ones(3)

        # initial infos
        self.init_quat = dict()
        self.init_pos = dict()

    def create_end_effector(self, gripper):
        gripper_names = [gripper] if isinstance(gripper, str) else gripper

        if gripper_names is None:
            self.end = None
        else:
            self.end: Dict[str, BaseEnd] = {}
            for agent, gripper in zip(self.agents, gripper_names):
                try:
                    self.end[agent] = robopal.robots.REGISTERED_ENDS[gripper](self.robot_data, self.robot_model, agent)
                except KeyError:
                    logging.error(f"End {gripper} is not registered. Available robots are {robopal.robots.REGISTERED_ENDS.keys()}.")
                    raise KeyError

    def build_from_xml(self, xml_path):
        self.robot_model = mujoco.MjModel.from_xml_path(filename=xml_path, assets=None)
        self.robot_data = mujoco.MjData(self.robot_model)
        # deepcopy for computing kinematics.
        self.kine_data: mujoco.MjData = copy.deepcopy(self.robot_data)

    def build_from_string(self, xml_string):
        self.robot_model = mujoco.MjModel.from_xml_string(xml_string)
        self.robot_data = mujoco.MjData(self.robot_model)
        # deepcopy for computing kinematics.
        self.kine_data: mujoco.MjData = copy.deepcopy(self.robot_data)

    @property
    def arm_joint_names(self) -> Dict[str, np.ndarray]:
        """ robot info """
        return self._arm_joint_names
    
    @property
    def arm_joint_indexes(self) -> Dict[str, np.ndarray]:
        """ robot info """
        return self._arm_joint_indexes
    
    @arm_joint_names.setter
    def arm_joint_names(self, names: Dict[str, np.ndarray]):
        self._arm_joint_names = names
        for agent, names in names.items():
            index = [mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            self._arm_joint_indexes[agent] = index
            
        self.mani_joint_bounds = {agent: (
            self.robot_model.jnt_range[self.arm_joint_indexes[agent], 0], 
            self.robot_model.jnt_range[self.arm_joint_indexes[agent], 1]
        ) for agent in self.agents}

    @property
    def arm_actuator_names(self) -> Dict[str, np.ndarray]:
        """ robot info """
        return self._arm_actuator_names
    
    @property
    def arm_actuator_indexes(self) -> Dict[str, np.ndarray]:
        """ robot info """
        return self._arm_actuator_indexes
    
    @arm_actuator_names.setter
    def arm_actuator_names(self, names: Dict[str, np.ndarray]):
        self._arm_actuator_names = names
        for agent, names in names.items():
            index = [mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
            self._arm_actuator_indexes[agent] = index

    @property
    def init_qpos(self) -> Dict[str, np.ndarray]:
        """ Robot's init joint position. """
        raise NotImplementedError

    @abc.abstractmethod
    def add_assets(self) -> None:
        """ Add objects into the xml file. """
        pass

    @property
    def jnt_num(self) -> Union[int, Dict[str, int]]:
        """ Number of joints. """
        return len(self.arm_joint_names[self.agents[0]])
    
    @property
    def name(self) -> str:
        """ Robot name. """
        return self.__class__.__name__

    def get_arm_qpos(self, agent: str = 'agent0') -> np.ndarray:
        """ Get arm joint position of the specified agent.

        :param agent: agent name
        :return: joint position
        """
        return np.array([self.robot_data.joint(j).qpos[0] for j in self.arm_joint_names[agent]])

    def get_arm_qvel(self, agent: str = 'agent0') -> np.ndarray:
        """ Get arm joint velocity of the specified agent.

        :param agent: agent name
        :return: joint position
        """
        return np.array([self.robot_data.joint(j).qvel[0] for j in self.arm_joint_names[agent]])

    def get_arm_qacc(self, agent: str = 'agent0') -> np.ndarray:
        """ Get arm joint accelerate of the specified agent.

        :param agent: agent name
        :return: joint position
        """
        return np.array([self.robot_data.joint(j).qacc[0] for j in self.arm_joint_names[agent]])
    
    def get_arm_tau(self, agent: str = 'agent0') -> np.ndarray:
        """ Get arm joint torque of the specified agent.

        :param agent: agent name
        :return: joint torque
        """
        return np.array([self.robot_data.actuator(a).ctrl[0] for a in self.arm_actuator_names[agent]])

    def get_mass_matrix(self, agent: str = 'agent0') -> np.ndarray:
        """ Get Mass Matrix
        ref https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/base_controller.py#L61

        :param agent: agent name
        :return: mass matrix
        """
        mass_matrix = np.ndarray(shape=(self.robot_model.nv, self.robot_model.nv), dtype=np.float64, order="C")
        # qM is inertia in joint space
        mujoco.mj_fullM(self.robot_model, mass_matrix, self.robot_data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.robot_data.qvel), len(self.robot_data.qvel)))
        return mass_matrix[self.arm_joint_indexes[agent], :][:, self.arm_joint_indexes[agent]]

    def get_coriolis_gravity_compensation(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.qfrc_bias[self.arm_joint_indexes[agent]]
    
    def get_end_xpos(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.body(self.end_name[agent]).xpos.copy()

    def get_end_xquat(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.body(self.end_name[agent]).xquat.copy()

    def get_end_xmat(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.body(self.end_name[agent]).xmat.copy().reshape(3, 3)
    
    def get_end_xvel(self, agent: str = 'agent0') -> np.ndarray:
        """ Computing the end effector velocity

        :param agent: agent name
        :return: end effector velocity, 6*1, [v, w]
        """
        return np.dot(self.get_full_jac(agent), self.get_arm_qvel(agent))

    def get_base_xpos(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.body(self.base_link_name[agent]).xpos.copy()

    def get_base_xquat(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.body(self.base_link_name[agent]).xquat.copy()

    def get_base_xmat(self, agent: str = 'agent0') -> np.ndarray:
        return self.robot_data.body(self.base_link_name[agent]).xmat.copy().reshape(3, 3)
    
    def get_full_jac(self, agent: str = 'agent0') -> np.ndarray:
        """ Computes the full model Jacobian, expressed in the coordinate world frame.

        :param agent: agent name
        :return: Jacobian
        """
        bid = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, self.end_name[agent])
        jacp = np.zeros((3, self.robot_model.nv))
        jacr = np.zeros((3, self.robot_model.nv))
        mujoco.mj_jacBody(self.robot_model, self.robot_data, jacp, jacr, bid)
        return np.concatenate([
            jacp[:, self.arm_joint_indexes[agent]], 
            jacr[:, self.arm_joint_indexes[agent]]
        ], axis=0).copy()
    
    def get_full_jac_pinv(self, agent: str = 'agent0') -> np.ndarray:
        """ Computes the full model Jacobian_pinv expressed in the coordinate world frame.

        :param agent: agent name
        :return: Jacobian_pinv
        """
        return np.linalg.pinv(self.get_full_jac(agent)).copy()
    
    def get_jac_dot(self, agent: str = 'agent0') -> np.ndarray:
        """ Computing the Jacobian_dot in the joint frame.
        https://github.com/google-deepmind/mujoco/issues/411#issuecomment-1211001685

        :param agent: agent name
        :return: Jacobian_dot
        """
        h = 1e-2
        J = self.get_full_jac(agent)

        original_qpos = self.robot_data.qpos.copy()
        mujoco.mj_integratePos(self.robot_model, self.robot_data.qpos, self.robot_data.qvel, h)
        mujoco.mj_comPos(self.robot_model, self.robot_data)
        mujoco.mj_kinematics(self.robot_model, self.robot_data)

        Jh = self.get_full_jac(agent)
        self.robot_data.qpos = original_qpos

        Jdot = (Jh - J) / h
        return Jdot
    