import os

from robopal.robots.base import *

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')


class DianaMed(BaseRobot):
    """ DianaMed robot base class. """
    def __init__(self,
                 scene='default',
                 manipulator='DianaMed',
                 gripper=None,
                 mount=None
                 ):
        super().__init__(
            name="diana_med",
            scene=scene,
            chassis=mount,
            manipulator=manipulator,
            gripper=gripper,
            g2m_body='0_link7',
            urdf_path=os.path.join(ASSET_DIR, "models/manipulators/DianaMed/DianaMed.urdf"),
        )
        self.arm_joint_names = {self.agents[0]: ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7']}
        self.arm_actuator_names = {self.agents[0]: ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7']}

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])}


class DualDianaMed(BaseRobot):
    """ Dual DianaMed robots base class. """
    def __init__(self,
                 scene='default',
                 manipulator=['DianaMed', 'DianaMed'],
                 gripper=['rethink_gripper', 'rethink_gripper'],
                 mount=['floor_left', 'floor_right'],
                 g2m_body=['0_link7', '1_link7']
                 ):
        super().__init__(
            name="diana_med",
            scene=scene,
            chassis=mount,
            manipulator=manipulator,
            gripper=gripper,
            g2m_body=g2m_body,
            urdf_path=os.path.join(ASSET_DIR, "models/manipulators/DianaMed/DianaMed.urdf"),
        )
        self.arm_joint_names = {self.agents[0]: ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7'],
                            self.agents[1]: ['1_j1', '1_j2', '1_j3', '1_j4', '1_j5', '1_j6', '1_j7']}
        self.arm_actuator_names = {self.agents[0]: ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7'],
                               self.agents[1]: ['1_a1', '1_a2', '1_a3', '1_a4', '1_a5', '1_a6', '1_a7']}

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0]),
                self.agents[1]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])}


class DianaAruco(DianaMed):
    def __init__(self):
        super().__init__(scene='default',
                         gripper='Swab_gripper', )

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/aruco/aruco.xml')


class DianaCollide(DianaMed):
    def add_assets(self):
        self.mjcf_generator.add_geom(node='worldbody', name='obstacle_box', pos='0.9 0.2 0.3',
                                     size='0.4 0.05 0.2', type='box')


class DianaCalib(DianaMed):
    """ DianaMed for Camera Calibration. """
    def add_assets(self):
        # link chessboard to the end
        self.mjcf_generator.add_texture('chessboard', type='2d',
                                        file=os.path.join(ASSET_DIR, 'textures/chessboard.png'))
        self.mjcf_generator.add_material('chessboard', texture='chessboard', texrepeat='1 1', texuniform='false')
        self.mjcf_generator.add_body(node='0_link7', name='chessboard')
        self.mjcf_generator.add_geom(node='chessboard', name='chessboard_box', pos='0.0 0 0.0', mass='0.001',
                                     euler="0 0 1.57", size='0.115 0.08 0.001', type='box', material='chessboard')

        # set realsense_d435
        self.mjcf_generator.add_mesh(name="cambase", file="objects/realsense_d435/meshes/cambase.STL")
        self.mjcf_generator.add_mesh(name="cam", file="objects/realsense_d435/meshes/cam.STL")

        cam = """<body pos="1.0 0.0 0.8" euler="0 0.785 3.14">
        <include file="objects/realsense_d435/realsense.xml"/>
    </body>"""
        self.mjcf_generator.add_node_from_str('worldbody', cam)


class DianaGrasp(DianaMed):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')

        goal_site = """<site name="goal_site" pos="0.4 0.0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.02167871, -0.16747492, 0.00730963, 2.5573341, -0.00401727, -0.42203728, -0.01099269])}


class DianaGraspMultiObjs(DianaGrasp):
    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/red_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'red_block', {'pos': '0.5 -0.1 0.46'})

        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'green_block', {'pos': '0.5 0.0 0.46'})

        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/blue_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'blue_block', {'pos': '0.5 0.1 0.46'})

        r_goal_site = """<site name="red_goal" pos="0.4 0.0 0.5" size="0.015 0.015 0.015" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', r_goal_site)

        g_goal_site = """<site name="green_goal" pos="0.4 0.0 0.5" size="0.015 0.015 0.015" rgba="0 1 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', g_goal_site)

        b_goal_site = """<site name="blue_goal" pos="0.4 0.0 0.5" size="0.015 0.015 0.015" rgba="0 0 1 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', b_goal_site)


class DianaDrawer(DianaMed):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        # add cupboard
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cupboard/cupboard.xml')

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.51198529, -0.44737435, -0.50879166, 2.3063219, 0.46514545, -0.48916244, -0.37233289])}


class DianaDrawerCube(DianaDrawer):
    def add_assets(self):
        super(DianaDrawerCube, self).add_assets()

        # add cube with random position
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')
        self.mjcf_generator.set_node_attrib('body', 'green_block', {'pos': '0.5 0.0 0.46'})

        goal_site = """<site name="cube_goal" pos="0.59 0.0 0.478" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.64551607, -0.29859465, -0.66478589, 2.3211311, 0.3205733, -0.61377277, -0.26366202])}


class DianaCabinet(DianaMed):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cabinet/cabinet.xml')
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cabinet/beam.xml')

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.71325374, 0.07279728, -0.72080385, 2.5239552, -0.07686951, -0.67930021, 0.05372948])}


class DianaAssemble(DianaMed):
    def __init__(self):
        super().__init__(scene='grasping',
                         gripper='rethink_gripper',
                         mount='top_point')

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/motor/motor.xml')
        for i in range(63):
            self.mjcf_generator.add_mesh(f'decomp{i}', ASSET_DIR + f'/objects/motor/output/decomp{i}.obj', scale='0.001 0.001 0.001')
            self.mjcf_generator.add_geom(node='cover', mesh=f'decomp{i}', type='mesh', group='1', friction="0.3 0.005 0.001")

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([-0.71325374, 0.07279728, -0.72080385, 2.5239552, -0.07686951, -0.67930021, 0.05372948])}
