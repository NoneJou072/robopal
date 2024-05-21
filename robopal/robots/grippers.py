import numpy as np

class BaseEnd(object):

    gripper_joint_names = dict()
    gripper_joint_indexes = dict()
    gripper_actuator_names = dict()
    gripper_actuator_indexes = dict()
    
    def __init__(self, robot_data) -> None:

        self.robot_data = robot_data

        self._ctrl_range = [-1, 1]

    def apply_action(self, action: np.ndarray) -> None:
        pass

    def open(self):
        self.apply_action(self._ctrl_range[1])

    def close(self):
        self.apply_action(self._ctrl_range[0])


class RethinkGripper(BaseEnd):
    def __init__(self, robot_data) -> None:
        super().__init__(robot_data)

        self._ctrl_range = [-0.01, 0.02]

    def apply_action(self, action):
        self.robot_data.actuator('0_gripper_l_finger_joint').ctrl[0] = action
        self.robot_data.actuator('0_gripper_r_finger_joint').ctrl[0] = action
        

class RobotiqGripper(BaseEnd):
    def __init__(self, robot_data) -> None:
        super().__init__(robot_data)

        self._ctrl_range = [0, 0.83]

    def apply_action(self, action):
        self.robot_data.actuator('0_robotiq_2f_85').ctrl[0] = action

    def open(self):
        self.apply_action(self._ctrl_range[0])

    def close(self):
        self.apply_action(self._ctrl_range[1])


class PandaHand(BaseEnd):
    def __init__(self, robot_data) -> None:
        super().__init__(robot_data)

        self._ctrl_range = [0, 255]

    def apply_action(self, action):
        self.robot_data.actuator('0_actuator8').ctrl[0] = action
