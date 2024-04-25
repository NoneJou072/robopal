import logging
import numpy as np
import mujoco
import pinocchio as pin

from robopal.envs.robot import RobotEnv
from robopal.robots.diana_med import DianaMed


class TestKinematics:
    def __init__(self):
        self.urdf_path = '../robopal/assets/models/manipulators/DianaMed/DianaMed.urdf'
        self.qpos = np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])
        self.env = RobotEnv(DianaMed(), render_mode=None)
        self.env.reset()

    def compute_jacobian_mojoco(self):
        jacpp = np.zeros((3, 7))
        jacrr = np.zeros((3, 7))
        i = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, '0_link7')
        mujoco.mj_jacBody(self.env.mj_model, self.env.mj_data, jacpp, jacrr, i)
        jac_mujoco = np.block([[jacpp], [jacrr]])

        print("-------------------------<mujoco>------------------------")
        print(jac_mujoco)

    def compute_jacobian_pinocchio(self):
        print("-------------------------<pinocchio>------------------------")
        print(self.env.kd_solver.get_full_jac(self.qpos))

    def get_joint_transform(self):
        print("-------------------------<joint transform>------------------------")
        pin.forwardKinematics(self.env.kd_solver.model, self.env.kd_solver.data, self.qpos)
        for i in range(7):
            print(self.env.kd_solver.data.oMi[i].translation)


if __name__ == '__main__':
    test = TestKinematics()
    # test.compute_jacobian_mojoco()
    # test.compute_jacobian_pinocchio()
    test.get_joint_transform()
