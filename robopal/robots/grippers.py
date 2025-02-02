import numpy as np

REGISTERED_ENDS = {}


class EndMetaClass(type):
    """Metaclass for registering robot arms"""

    def __new__(meta, name, bases, attrs):
        cls = super().__new__(meta, name, bases, attrs)

        if not cls.__name__ == "BaseEnd":
            REGISTERED_ENDS[cls.__name__] = cls
        return cls


class BaseEnd(object, metaclass=EndMetaClass):

    gripper_joint_names = dict()
    gripper_joint_indexes = dict()
    gripper_actuator_names = dict()
    gripper_actuator_indexes = dict()

    # dt will pass in after the environment is instantiated
    dt: int = None

    _ctrl_range = [-1, 1]

    def __init__(self, robot_data, robot_model, agent) -> None:

        self.robot_data = robot_data
        self.robot_model = robot_model
        self.agent_id = agent[-1]

    def apply_action(self, action: np.ndarray) -> None:
        pass

    def open(self):
        self.apply_action(self._ctrl_range[1])

    def close(self):
        self.apply_action(self._ctrl_range[0])

    def get_finger_observations(self):
        pass

    def reset(self):
        pass


class RethinkGripper(BaseEnd):

    _ctrl_range = [-0.01, 0.02]

    def apply_action(self, action):
        self.robot_data.actuator(f'{self.agent_id}_gripper_l_finger_joint').ctrl[0] = action
        self.robot_data.actuator(f'{self.agent_id}_gripper_r_finger_joint').ctrl[0] = action
    
    def get_finger_observations(self):
        return np.concatenate([
            self.robot_data.joint(f'{self.agent_id}_l_finger_joint').qpos,
            self.robot_data.joint(f'{self.agent_id}_l_finger_joint').qvel * self.dt
        ], axis=0)


class RobotiqGripper(BaseEnd):

    _ctrl_range = [0, 0.83]

    def apply_action(self, action):
        self.robot_data.actuator(f'{self.agent_id}_robotiq_2f_85').ctrl[0] = action

    def open(self):
        self.apply_action(self._ctrl_range[0])

    def close(self):
        self.apply_action(self._ctrl_range[1])


class PandaHand(BaseEnd):

    _ctrl_range = [0, 255]

    def apply_action(self, action):
        self.robot_data.actuator(f'{self.agent_id}_actuator8').ctrl[0] = action

    def get_finger_observations(self):
        return np.concatenate([
            self.robot_data.joint(f'{self.agent_id}_finger_joint1').qpos,
            self.robot_data.joint(f'{self.agent_id}_finger_joint1').qvel * self.dt
        ], axis=0)
