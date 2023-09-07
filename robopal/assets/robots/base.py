import mujoco
import numpy as np
from robopal.commons import RobotGenerator


class ArmBase(object):
    def __init__(self,
                 name: str = None,
                 scene: str = 'default',
                 chassis: str = 'default',
                 manipulator: str = None,
                 gripper: str = None,
                 g2m_body: list = None,
                 urdf_path: str = None,
                 xml_path: str = None,
                 ):
        self.name = name
        self.robot = RobotGenerator(
            scene=scene,
            chassis=chassis,
            manipulator=manipulator,
            gripper=gripper,
            g2m_body=g2m_body
        )
        self.add_assets()

        self.urdf_path = urdf_path
        if isinstance(xml_path, str):
            self.xml_path =xml_path
        else:
            self.xml_path = self.robot.xml

        self.robot_model = mujoco.MjModel.from_xml_path(filename=self.xml_path, assets=None)
        self.robot_data = mujoco.MjData(self.robot_model)
        self.arm = []

    def add_assets(self):
        pass

    class Arm:
        """
        internal class
        using for precise arm state, double (left and right) or single
        """

        def __init__(self, ArmBase, state):
            self.robot_model = ArmBase.robot_model
            self.robot_data = ArmBase.robot_data
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
