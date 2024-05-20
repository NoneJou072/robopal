import os

from robopal.robots.base import *

ASSET_DIR = os.path.join(os.path.dirname(__file__), '../assets')


class DualDianaMed(BaseRobot):
    """ Dual DianaMed robots base class. """
    def __init__(self,
                 scene='default',
                 manipulator=['DianaMed', 'DianaMed'],
                 gripper=['rethink_gripper', 'rethink_gripper'],
                 mount=['floor_left', 'floor_right'],
                 attached_body=['0_attachment', '1_attachment']
                 ):
        super().__init__(
            name="diana_med",
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            attached_body=attached_body,
        )
        self.arm_joint_names = {self.agents[0]: ['0_j1', '0_j2', '0_j3', '0_j4', '0_j5', '0_j6', '0_j7'],
                            self.agents[1]: ['1_j1', '1_j2', '1_j3', '1_j4', '1_j5', '1_j6', '1_j7']}
        self.arm_actuator_names = {self.agents[0]: ['0_a1', '0_a2', '0_a3', '0_a4', '0_a5', '0_a6', '0_a7'],
                               self.agents[1]: ['1_a1', '1_a2', '1_a3', '1_a4', '1_a5', '1_a6', '1_a7']}
        self.base_link_name = {self.agents[0]: '0_base_link', self.agents[1]: '1_base_link'}
        self.end_name = {self.agents[0]: '0_link7', self.agents[1]: '1_link7'}

    @property
    def init_qpos(self):
        """ Robot's init joint position. """
        return {self.agents[0]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0]),
                self.agents[1]: np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.00, np.pi / 4.0, 0.0])}


class DualDianaGrasp(DualDianaMed):
    def __init__(self):
        super().__init__(
            scene='grasping',
            manipulator=['DianaMed', 'DianaMed'],
            gripper=['rethink_gripper', 'rethink_gripper'],
            mount=['cylinder', 'cylinder2'],
            attached_body=['0_attachment', '1_attachment']
        )

    def add_assets(self):
        self.mjcf_generator.add_node_from_xml(ASSET_DIR + '/objects/cube/green_cube.xml')

        goal_site = """<site name="goal_site" pos="0.4 0.0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere" />"""
        self.mjcf_generator.add_node_from_str('worldbody', goal_site)

        # set mount pose.
        self.mjcf_generator.set_node_attrib('body', '0_mount_base_link', {'pos': '1.2 0.0 0.3', 'quat': "0 0 0 1"})
        self.mjcf_generator.set_node_attrib('body', '1_mount_base_link', {'pos': '-0.4 0.0 0.3', 'quat': "1 0 0 0"})

    @property
    def init_qpos(self):
        return {self.agents[0]: np.array([0, 0.1, 0, 2.28, 0, -1, 0]),
                self.agents[1]: np.array([0, 0.1, 0, 2.28, 0, -1, 0])}
    