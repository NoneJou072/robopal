from robopal.assets.robots.base import *
import os


class DianaMedBase(ArmBase):
    def __init__(self,
                 scene='default',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            name="diana_med",
            scene=scene,
            chassis=mount,
            manipulator='DianaMed',
            gripper=gripper,
            g2m_body=['0_link7'],
            urdf_path=os.path.join(os.path.dirname(__file__), "../models/manipulators/DianaMed/DianaMed.urdf"),
        )
        self.single_arm = self.Arm(self, 'single')
        self.single_arm.joint_index = ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7']
        self.single_arm.actuator_index = ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7']
        self.single_arm.setArmInitPose(self.init_qpos)
        self.arm.append(self.single_arm)

        self.jnt_num = self.single_arm.jnt_num

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])


class DianaMed(DianaMedBase):
    def __init__(self):
        super().__init__(scene='default',
                         gripper=None, )


class DianaAruco(DianaMedBase):
    def __init__(self):
        super().__init__(scene='visual_imped',
                         gripper='Swab_gripper', )

    def add_assets(self):
        self.robot.add_texture('aruco', type='2d',
                               file=os.path.join(os.path.dirname(__file__), '../textures/aruco.png'))
        self.robot.add_material('aruco', texture='aruco', texrepeat='1 1', texuniform='false')
        self.robot.add_body(node='worldbody', name='aruco')
        self.robot.add_geom(node='aruco', name='aruco_box', pos='0.919 0 1.27', mass='0.001',
                            euler="0 -1.57 0", size='0.05 0.05 0.001', type='box', material='aruco')
        self.robot.add_joint(node='aruco', name='aruco_x', type='slide', axis='1 0 0')
        self.robot.add_joint(node='aruco', name='aruco_y', type='slide', axis='0 1 0')
        self.robot.add_joint(node='aruco', name='aruco_z', type='slide', axis='0 0 1')


class DianaGrasp(DianaMedBase):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.00985161, -0.71512797,  0.00479528,  1.59160709, -0.00473849, -0.83985286, -0.00324085])
