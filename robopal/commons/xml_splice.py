import os
import inspect
from typing import Union, List, Iterable
import xml.etree.ElementTree as ET
from os import path
from copy import deepcopy
import logging

import robopal

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
ASSETS_PATH = os.path.join(ROBOPAL_PATH, 'assets')
MODELS_PATH = os.path.join(ASSETS_PATH, 'models')
MOUNTS_DIR_PATH = os.path.join(MODELS_PATH, 'mounts')
GRIPPERS_DIR_PATH = os.path.join(MODELS_PATH, 'grippers')
MANIPULATORS_DIR_PATH = os.path.join(MODELS_PATH, 'manipulators')
SCENES_DIR_PATH = os.path.join(ASSETS_PATH, 'scenes')


class RobotGenerator(object):
    """ XML Splicer class for splicing the robot with the given scene, mount, manipulator and gripper.
    
    :param name: robot name
    :param scene: scene name
    :param mount: mount name
    :param manipulator: manipulator name
    :param gripper: gripper name
    :param kwargs: other arguments
    """
    def __init__(self,
                 scene: str = 'default',
                 mount: Union[str, List[str]] = None,
                 manipulator: Union[str, List[str]] = None,
                 gripper: Union[str, List[str]] = None,
                 **kwargs,
                 ):
        
        self._mjcf_path = None
        if 'xml_path' in kwargs and isinstance(kwargs['xml_path'], str):
            self._mjcf_path = path.abspath(kwargs["xml_path"])

        self.tree = None
        self.root = None

        self._concatenate_robot(
            scene=scene,
            mount=mount,
            manipulator=manipulator,
            gripper=gripper,
            **kwargs,
        )

    def _concatenate_robot(
            self, 
            scene=None, 
            mount=None, 
            manipulator=None, 
            gripper=None, 
            **kwargs
        ):
        """ Splice the robot with the given scene, mount, manipulator and gripper.

        :param scene: scene name
        :param mount: mount name
        :param manipulator: manipulator name
        :param gripper: gripper name
        :param kwargs:
        """
        # Check the scene.
        if isinstance(scene, str):
            if scene.endswith('.xml'):
                scene_path = scene
            else:
                scene_path = path.join(SCENES_DIR_PATH, '{}.xml'.format(scene))
            self._init_scene(scene_path)
        else:
            raise ValueError("Must have scene.xml to generate the world.")

        if isinstance(mount, str):
            mount = [mount]
        if isinstance(manipulator, str):
            manipulator = [manipulator]
        if isinstance(gripper, str):
            gripper = [gripper]

        if isinstance(mount, list):
            for ch_id, ch_name in enumerate(mount):
                if ch_name.endswith('.xml'):
                    mount_path = ch_name
                else:
                    mount_path = path.join(MOUNTS_DIR_PATH, ch_name, '{}.xml'.format(ch_name))
                self.add_all_component_from_xml(
                    mount_path, 
                    goal_body = (ch_id, 'worldbody'),
                    is_rename_tag = not (len(mount) == 1 and len(manipulator) > 1)
                )

        if isinstance(manipulator, Iterable):
            for mani_id, mani_name in enumerate(manipulator):
                if mani_name.endswith('.xml'):
                    manipulator_path = mani_name
                else:
                    manipulator_path = path.join(MANIPULATORS_DIR_PATH, mani_name, '{}.xml'.format(mani_name))
                self.add_all_component_from_xml(manipulator_path,
                                            goal_body=(mani_id, f'{mani_id}_mount_base_link') if mount is not None else (mani_id, 'worldbody'))

            if gripper is not None:
                assert kwargs['attached_body'] is not None, "Please specify the attached_body for the gripper."
                attached_body = kwargs['attached_body']
                if isinstance(kwargs['attached_body'], str):
                    attached_body = [attached_body]
                logging.info(f"attached_body: {attached_body}, type: {type(attached_body)}")
                if isinstance(gripper, Iterable):
                    for goal_body, g in zip(enumerate(attached_body), gripper):
                        gripper_path = path.join(GRIPPERS_DIR_PATH, g, '{}.xml'.format(g))
                        self.add_all_component_from_xml(gripper_path, goal_body=goal_body)
                        logging.info(f"Add gripper {g} to the {goal_body}.")

    def _init_scene(self, scene):
        """
        Initialize the scene xml for the given scene file. We build the tree here
        as the global tree. Each other node will append to it.
        :param scene: xml file path of scene
        """
        self.tree = ET.parse(scene)
        self.root = self.tree.getroot()
        ELEMENTS = ['asset', 'worldbody', 'actuator', 'default', 'contact', 'sensor', 'equality', 'tendon']
        for element in ELEMENTS:
            if self.root.find(element) is None:
                self.root.append(ET.Element(element))
            else:
                for node in self.root.findall(element):
                    self._rename_path(scene, node)

    def add_all_component_from_xml(self, xml: str, goal_body: tuple, is_rename_tag=True) -> None:
        """
        For each input xml file, we extract the component we need, and append it into global tree.

        :param xml: xml file path
        :param goal_body: goal body, a tuple consist of id and name
        """
        if not isinstance(xml, str):
            raise ValueError("Please checkout your xml path.")
        tree = ET.parse(xml)
        root_node = tree.getroot()
        # preprocess
        if is_rename_tag:
            self._rename_tag(goal_body[0], root_node)

        # add assets
        if root_node.find('asset') is not None:
            self._rename_path(xml, root_node.find('asset'))
            for asset in root_node.find('asset'):
                # check if the asset is already in the global tree
                if 'name' in asset.attrib and \
                self.root.find(f'.//material[@name=\'{asset.attrib["name"]}\']') is not None:
                    continue
                # check if the mesh is already in the global tree
                if 'file' in asset.attrib and 'name' not in asset.attrib and \
                self.root.find(f'.//mesh[@file=\'{asset.attrib["file"]}\']') is not None:
                    continue
                self.root.find('asset').append(asset)
        # add world-body
        if root_node.find('worldbody') is not None:
            for body in root_node.find('worldbody'):
                node = self.root.find(f'.//body[@name=\'{goal_body[1]}\']')
                if node is None:
                    node = self.root.find('worldbody')
                node.append(body)
        # add actuator
        if root_node.find('actuator') is not None:
            for actuator in root_node.find('actuator'):
                self.root.find('actuator').append(actuator)
        # add default
        if root_node.find('default') is not None:
            for default in root_node.find('default'):
                # check if the default class is already in the global tree
                if self.root.find(f'.//default[@class=\'{default.attrib["class"]}\']') is not None:
                    continue
                self.root.find('default').append(default)
        # add contact
        if root_node.find('contact') is not None:
            for contact in root_node.find('contact'):
                self.root.find('contact').append(contact)
        # add sensor
        if root_node.find('sensor') is not None:
            for sensor in root_node.find('sensor'):
                self.root.find('sensor').append(sensor)
        # add equality
        if root_node.find('equality') is not None:
            for equality in root_node.find('equality'):
                self.root.find('equality').append(equality)
        # add tendon
        if root_node.find('tendon') is not None:
            for tendon in root_node.find('tendon'):
                self.root.find('tendon').append(tendon)

    def _rename_path(self, xml_path, asset_node: ET.Element):
        """ Reset file path of the assets to abstract path.
        Note all the texture files should put into the texture dict.
        :param xml_path: path of specified xml file
        :param asset_node: node
        """
        for child in asset_node:
            if child.tag == 'mesh' and child.attrib['file'] is not None:
                child.attrib['file'] = path.join(path.dirname(xml_path), child.attrib['file'])
            elif child.tag == 'texture' and 'file' in child.attrib:
                TEXTURE_DIR_PATH = path.dirname(__file__) + '/../assets/textures'
                child.attrib['file'] = path.join(TEXTURE_DIR_PATH,
                                                 child.attrib['file'])

    def _rename_tag(self, id, node: ET.Element):
        """ add indice to the tag name
        """
        for mesh in node.findall('.//mesh[@name]'):
            for element in node.findall('.//geom[@mesh=\'{}\']'.format(mesh.attrib['name'])):
                element.attrib['mesh'] = '{}_{}'.format(id, element.attrib['mesh'])
            mesh.attrib['name'] = '{}_{}'.format(id, mesh.attrib['name'])

        # find all nodes according the specified name
        for body in node.findall('.//body[@name]'):
            body_name = deepcopy(body.attrib['name'])
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == body_name and key != 'mesh':
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])

        # find all geom nodes with name attribute
        for geom in node.findall('.//geom[@name]'):
            geom.attrib['name'] = '{}_{}'.format(id, geom.attrib['name'])

        # find all joint nodes with name attribute
        for joint in node.findall('.//joint[@name]'):
            joint_name = deepcopy(joint.attrib['name'])
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == joint_name:
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])

        # find all connect nodes with name attribute
        for connect in node.findall('.//connect[@name]'):
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == connect.attrib['name']:
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])

        # find all contact nodes with name attribute
        # contact body has already renamed above.
        pass 

        # find all site nodes with name attribute
        for site in node.findall('.//site[@name]'):
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == site.attrib['name'] and element.tag != 'site':
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])
            site.set('name', '{}_{}'.format(id, site.attrib['name']))

        for camera in node.findall('.//camera[@name]'):
            camera.set('name', '{}_{}'.format(id, camera.attrib['name']))

        for sensor in node.findall('.//sensor/*[@name]'):
            sensor.set('name', '{}_{}'.format(id, sensor.attrib['name']))

        for actuator in node.findall('.//actuator/*[@name]'):
            actuator.set('name', '{}_{}'.format(id, actuator.attrib['name']))

        for tendon in node.findall('.//tendon/*[@name]'):
            tendon.set('name', '{}_{}'.format(id, tendon.attrib['name']))
        # find all nodes with tendon attribute, then rename them
        for node in node.findall('.//*[@tendon]'):
            node.attrib['tendon'] = '{}_{}'.format(id, node.attrib['tendon'])

    def add_node_from_xml(
            self, 
            xml_path: str = None, 
            parent_body_name: str = None
        ):
        """ Add node from xml file. The attached node is the parent node of the new node.
        :param xml_path: the path of the xml file.
        :param node: the name of the linked body. if set None, will add to the worldbody.
        """
        assert isinstance(xml_path, str), "Please checkout your xml path."

        sub_tree = ET.parse(xml_path)

        for node_type in ['worldbody', 'asset', 'actuator', 'default']:
            # find the sub node
            sub_node = sub_tree.getroot().find(node_type)
            if sub_node is not None:
                if node_type == 'worldbody' and parent_body_name is not None:
                    parent_node = self.root.find(f'.//body[@name=\'{parent_body_name}\']')
                else:
                    parent_node = self.root.find(node_type)
                
                # get all sub nodes in specified node
                sub_node = sub_node.findall('*')
                # add all sub nodes to the parent node
                if isinstance(sub_node, list):
                    for node in sub_node:
                        parent_node.append(node)
                else:
                    parent_node.append(sub_node)

    def add_node_from_str(self, father_node: str, xml_text: str):
        parent_element = self.root.find(father_node)
        new_element = ET.fromstring(xml_text)
        parent_element.append(new_element)

    def set_node_attrib(self, node: str, name: str, attrib: dict):
        """ Set node attribute.
        e.g.
        >>> self.set_node_attrib('body', 'green_block', {'pos': '0.5 0.0 0.46'})
        :param node: node name
        :param attrib: attribute dict
        """
        node_element = self.root.find(f'.//{node}[@name=\'{name}\']')
        for key in attrib:
            node_element.set(key, attrib[key])

    def add_texture(self, name: str, type: str, file: str):
        parent_element = self.root.find("asset")
        texture_element = ET.Element('texture')
        texture_element.set('name', name)
        texture_element.set('type', type)
        texture_element.set('file', file)
        parent_element.append(texture_element)

    def add_material(self, name: str, texture: str, texrepeat: str, texuniform: str):
        parent_element = self.root.find("asset")
        material_element = ET.Element('material')
        material_element.set('name', name)
        material_element.set('texture', texture)
        material_element.set('texrepeat', texrepeat)
        material_element.set('texuniform', texuniform)
        parent_element.append(material_element)

    def add_mesh(self, name: str, file: str, **kwargs):
        parent_element = self.root.find("asset")
        mesh_element = ET.Element('mesh')
        mesh_element.set('name', name)
        mesh_element.set('file', file)
        for key in kwargs:
            mesh_element.set(key, kwargs[key])
        parent_element.append(mesh_element)

    def add_geom(self, node: str, **kwargs):
        if node == 'worldbody':
            geom_element = self.root.find('worldbody')
        else:
            geom_element = self.root.find(f'.//body[@name=\'{node}\']')
        geom = ET.Element('geom')
        for key in kwargs:
            geom.set(key, kwargs[key])
        geom_element.append(geom)

    def add_body(self, node: str, **kwargs):
        if node == 'worldbody':
            body_element = self.root.find('worldbody')
        else:
            body_element = self.root.find(f'.//body[@name=\'{node}\']')
        body = ET.Element('body')
        for key in kwargs:
            body.set(key, kwargs[key])
        body_element.append(body)

    def add_joint(self, node: str, **kwargs):
        if node == 'worldbody':
            joint_element = self.root.find('worldbody')
        else:
            joint_element = self.root.find(f'.//body[@name=\'{node}\']')
        joint = ET.Element('joint')
        for key in kwargs:
            joint.set(key, kwargs[key])
        joint_element.append(joint)

    def save_xml(self, xml_name = "robot"):
        """ Save xml file with identity path"""
        self._mjcf_path = path.abspath(path.join(ASSETS_PATH, f"{xml_name}.xml"))
        self.tree.write(self._mjcf_path)
        logging.info(f"mjcf model has been saved to {self._mjcf_path}")
        return self._mjcf_path

    def get_xml_path(self):
        """ Load xml file"""
        return self._mjcf_path
