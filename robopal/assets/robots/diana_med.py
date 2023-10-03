import os
from abc import ABC

from robopal.assets.robots.base import *
import numpy as np


class DianaMedBase(BaseArm, ABC):
    """ DianaMed robot base class. """
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
        self.single_arm = self.PartArm(self, 'single')
        self.single_arm.joint_index = ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7']
        self.single_arm.actuator_index = ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7']
        self.single_arm.setArmInitPose(self.init_qpos)
        self.arm.append(self.single_arm)

        self.jnt_num = self.single_arm.jnt_num

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])


class DianaMed(DianaMedBase, ABC):
    """ DianaMed robot class. """
    def __init__(self):
        super().__init__(scene='default',
                         gripper=None, )


class DianaAruco(DianaMedBase):
    """ DianaMed robot class. """
    def __init__(self):
        super().__init__(scene='visual_imped',
                         gripper='Swab_gripper', )

    def add_assets(self):
        self.mjcf_generator.add_texture('aruco', type='2d',
                               file=os.path.join(os.path.dirname(__file__), '../textures/aruco.png'))
        self.mjcf_generator.add_material('aruco', texture='aruco', texrepeat='1 1', texuniform='false')
        self.mjcf_generator.add_body(node='worldbody', name='aruco')
        self.mjcf_generator.add_geom(node='aruco', name='aruco_box', pos='0.919 0 1.27', mass='0.001',
                            euler="0 -1.57 0", size='0.05 0.05 0.001', type='box', material='aruco')
        self.mjcf_generator.add_joint(node='aruco', name='aruco_x', type='slide', axis='1 0 0')
        self.mjcf_generator.add_joint(node='aruco', name='aruco_y', type='slide', axis='0 1 0')
        self.mjcf_generator.add_joint(node='aruco', name='aruco_z', type='slide', axis='0 0 1')


class DianaGrasp(DianaMedBase, ABC):
    """ DianaMed robot class. """
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        random_x_pos = np.random.uniform(0.35, 0.55)
        random_y_pos = np.random.uniform(-0.15, 0.15)
        block = f"""<body pos="{random_x_pos} {random_y_pos} {0.46}" name="green_block">
            <joint name="object2:joint" type="free" damping="0.01"/>
            <geom name="green_block" size="0.02 0.02 0.02" rgba="0 1 0 1" type="box" solimp="0.998 0.998 0.001" group="1" condim="4" mass="0.01"/>
            <site name="green_block" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />
        </body>"""
        self.mjcf_generator.add_node_from_str('worldbody', block)

        random_goal_x_pos = np.random.uniform(0.35, 0.55)
        random_goal_y_pos = np.random.uniform(-0.15, 0.15)
        random_goal_z_pos = np.random.uniform(0.46, 0.66)

        block_pos = np.array([random_x_pos, random_y_pos, 0.46])
        goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        while np.linalg.norm(block_pos - goal_pos) <= 0.05:
            random_goal_x_pos = np.random.uniform(0.4, 0.6)
            random_goal_y_pos = np.random.uniform(-0.2, 0.2)
            random_goal_z_pos = np.random.uniform(0.45, 0.66)
            goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])

        goal_site = f"""<body pos="{random_goal_x_pos} {random_goal_y_pos} {random_goal_z_pos}" name="goal_site">
                    <site name="goal_site" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />
                </body>"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([0.02167871, -0.16747492, 0.00730963, 2.5573341, -0.00401727, -0.42203728, -0.01099269])


class DianaDrawer(DianaMedBase, ABC):
    """ DianaMed robot class. """
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        # add cupboard with fixed position
        self.mjcf_generator.add_mesh('cupboard', 'objects/cupboard/cupboard.stl', scale='0.001 0.001 0.001')
        self.mjcf_generator.add_mesh('drawer_up', 'objects/cupboard/drawer_up.stl', scale='0.001 0.001 0.001')
        self.mjcf_generator.add_mesh('drawer_down', 'objects/cupboard/drawer_down.stl', scale='0.001 0.001 0.001')
        cupboard_x_pos = 0.66
        cupboard_y_pos = 0.0
        self.mjcf_generator.add_node_from_xml('worldbody', os.path.dirname(__file__) + '/../objects/cupboard/cupboard.xml')
        self.mjcf_generator.set_node_attrib('cupboard', {'pos': f'{cupboard_x_pos} {cupboard_y_pos} {0.42}'})

        # add goal site with random position
        random_goal_x_pos = np.random.uniform(0.4, 0.6)
        random_goal_z_pos = np.random.uniform(0.45, 0.66)
        goal_site = f"""<body pos="{0.5} {0.0} {0.5}" name="goal_site">
                    <site name="goal_site" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />
                </body>"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([-6.04886299e-01, -3.95403466e-01, -6.30041490e-01, 2.16857024e+00, 3.65813627e-01, -6.86444384e-01, -2.76880621e-01])


class DianaDrawerCube(DianaMedBase, ABC):
    """ DianaMed robot class. """
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        self.mjcf_generator.add_mesh('cupboard', 'objects/cupboard/cupboard.stl', scale='0.001 0.001 0.001')
        self.mjcf_generator.add_mesh('drawer_up', 'objects/cupboard/drawer_up.stl', scale='0.001 0.001 0.001')
        self.mjcf_generator.add_mesh('drawer_down', 'objects/cupboard/drawer_down.stl', scale='0.001 0.001 0.001')
        cupboard_x_pos = 0.66
        cupboard_y_pos = 0.0
        self.mjcf_generator.add_node_from_xml('worldbody', os.path.dirname(__file__) + '/../objects/cupboard/cupboard.xml')
        self.mjcf_generator.set_node_attrib('cupboard', {'pos': f'{cupboard_x_pos} {cupboard_y_pos} {0.42}'})

        # add cube with random position
        random_x_pos = np.random.uniform(0.4, 0.6)
        random_y_pos = np.random.uniform(-0.2, 0.2)
        self.mjcf_generator.add_node_from_xml('worldbody', os.path.dirname(__file__) + '/../objects/cube/cube.xml')
        self.mjcf_generator.set_node_attrib('green_block', {'pos': f'{random_x_pos} {random_y_pos} {0.46}'})

        # add goal site with random position
        random_goal_x_pos = np.random.uniform(0.4, 0.6)
        random_goal_y_pos = np.random.uniform(-0.2, 0.2)
        random_goal_z_pos = np.random.uniform(0.45, 0.66)

        cupboard_pos = np.array([cupboard_x_pos, cupboard_y_pos, 0.46])
        goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])
        while np.linalg.norm(cupboard_pos - goal_pos) <= 0.1:
            random_goal_x_pos = np.random.uniform(0.4, 0.6)
            random_goal_y_pos = np.random.uniform(-0.2, 0.2)
            random_goal_z_pos = np.random.uniform(0.45, 0.66)
            goal_pos = np.array([random_goal_x_pos, random_goal_y_pos, random_goal_z_pos])

        goal_site = f"""<body pos="{random_goal_x_pos} {random_goal_y_pos} {random_goal_z_pos}" name="goal_site">
                    <site name="goal_site" pos="0 0 0" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />
                </body>"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return np.array([-6.04886299e-01, -3.95403466e-01, -6.30041490e-01, 2.16857024e+00, 3.65813627e-01, -6.86444384e-01, -2.76880621e-01])
