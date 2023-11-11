import logging
import numpy as np
import mujoco
try:
    from robopal.envs.jnt_ctrl_env import JntCtrlEnv
    from robopal.robots.diana_med import DianaMed
except ImportError:
    pass
try:
    from KDL_utils import KDL_utils
except ImportError:
    pass


class TestKinematics:
    def __init__(self):
        self.urdf_path = '../robopal/assets/models/manipulators/DianaMed/DianaMed.urdf'
        self.qpos = np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])
        self.env = JntCtrlEnv(DianaMed(), is_render=False)
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
        print(self.env.kdl_solver.get_full_jac(self.qpos))

    def compute_jacobian_kdl(self):
        self.kdl_solver = KDL_utils(self.urdf_path)
        print("-------------------------<kdl>------------------------")
        print(self.kdl_solver.getJac(self.qpos))


if __name__ == '__main__':
    test = TestKinematics()
    test.compute_jacobian_mojoco()
    test.compute_jacobian_pinocchio()
