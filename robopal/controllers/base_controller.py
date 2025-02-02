from typing import Union, Dict

import mujoco
import numpy as np
import robopal.commons.transform as T
from robopal.robots.base import BaseRobot


class BaseController(object):
    reference = 'base'

    def __init__(self, robot: BaseRobot) -> None:
        self.name = None
        self.robot = robot
        self.dofs = robot.jnt_num

        assert self.reference in ['base', 'world'], "Invalid reference frame."

    def step_controller(
            self, action: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """ Step once computing for all agents.
        """
        raise NotImplementedError

    def reset(self):
        """ reset controller. """
        pass

    def forward_kinematics(self, q, agent='agent0'):

        def set_joint_qpos(qpos: np.ndarray, agent: str = 'agent0'):
            """ Set joint position. """
            assert qpos.shape[0] == self.robot.jnt_num
            for j, per_arm_joint_names in enumerate(self.robot.arm_joint_names[agent]):
                self.robot.kine_data.joint(per_arm_joint_names).qpos = qpos[j]

        set_joint_qpos(q, agent)

        mujoco.mj_fwdPosition(self.robot.robot_model, self.robot.kine_data)

        base_pos = self.robot.kine_data.body(self.robot.base_link_name[agent]).xpos
        base_mat = self.robot.kine_data.body(self.robot.base_link_name[agent]).xmat.reshape(3, 3)
        end_pos = self.robot.kine_data.body(self.robot.end_name[agent]).xpos
        end_mat = self.robot.kine_data.body(self.robot.end_name[agent]).xmat.reshape(3, 3)

        end = T.make_transform(end_pos, end_mat)
        base = T.make_transform(base_pos, base_mat)

        if self.reference == 'base':
            ret = np.linalg.pinv(base) @ end
            return ret[:3, -1], T.mat_2_quat(ret[:3, :3])
        else:
            return end_pos, T.mat_2_quat(end_mat)
