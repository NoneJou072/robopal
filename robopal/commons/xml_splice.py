import xml.etree.ElementTree as ET
from os import path

ASSETS_PATH = path.join(path.dirname(path.dirname(__file__)), 'assets')
MODELS_PATH = path.join(ASSETS_PATH, 'models')
CHASSISES_DIR_PATH = path.join(MODELS_PATH, 'mounts')
GRIPPERS_DIR_PATH = path.join(MODELS_PATH, 'grippers')
MANIPULATORS_DIR_PATH = path.join(MODELS_PATH, 'manipulators')
SCENES_DIR_PATH = path.join(ASSETS_PATH, 'scenes')


class XMLSplicer:
    def __init__(self,
                 name='robot',
                 scene='default',
                 chassis=None,
                 manipulator=None,
                 gripper=None,
                 **kwargs,
                 ):
        self.xml_name = name
        self.tree = None
        self.root = None
        self.splice_robot(
            scene=scene,
            chassis=chassis,
            manipulator=manipulator,
            gripper=gripper,
            **kwargs,
        )

    def _init_scene(self, scene):
        """
        Initialize the scene xml for the given scene file. We build the tree here
        as the global tree. Each other node will append to it.
        :param scene: xml file path of scene
        """
        self.tree = ET.parse(scene)
        self.root = self.tree.getroot()
        ELEMENTS = ['asset', 'worldbody', 'actuator', 'default', 'contact', 'sensor', 'equality']
        for element in ELEMENTS:
            if self.root.find(element) is None:
                self.root.append(ET.Element(element))
            else:
                for node in self.root.findall(element):
                    self._file_repath(scene, node)

    def add_component_from_xml(self, xml: str, goal_body: tuple):
        """
        For each input xml file, we extract the component we need, and append it into
        global tree.
        :param xml: xml file path
        :param goal_body: goal body, a tuple consist of id and name
        """
        if not isinstance(xml, str):
            raise ValueError("Please checkout your xml path.")
        tree = ET.parse(xml)
        root_node = tree.getroot()
        # preprocess
        self.tag_rename(goal_body[0], root_node)

        # add assets
        if root_node.find('asset') is not None:
            self._file_repath(xml, root_node.find('asset'))
            for asset in root_node.find('asset'):
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

    def _file_repath(self, xml_path, asset_node: ET.Element):
        """ Reset file path of the assets to abstract path.
        Note all the texture files should put into the texture dict.
        :param xml_path: path of specified xml file
        :param asset_node: node
        """
        for child in asset_node:
            if child.tag == 'mesh' and child.attrib['file'] is not None:
                child.attrib['file'] = path.join(path.dirname(xml_path), child.attrib['file'])
            elif child.tag == 'texture' and 'file' in child.attrib:
                child.attrib['file'] = path.join(path.dirname(path.dirname(xml_path)), 'textures',
                                                 path.basename(child.attrib['file']))

    def tag_rename(self, id, node: ET.Element):
        for mesh in node.findall('.//mesh[@name]'):
            for element in node.findall('.//geom[@mesh=\'{}\']'.format(mesh.attrib['name'])):
                element.attrib['mesh'] = '{}_{}'.format(id, element.attrib['mesh'])
            mesh.attrib['name'] = '{}_{}'.format(id, mesh.attrib['name'])

        for body in node.findall('.//body[@name]'):
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == body.attrib['name'] and key != 'mesh':
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])

        for default in node.findall('.//default'):
            if 'class' in default.attrib:
                for geom in node.findall('.//geom'):
                    if 'class' in geom.attrib and geom.get('class') == default.attrib['class']:
                        geom.set('class', '{}_{}'.format(id, geom.attrib['class']))
                for body in node.findall('.//body'):
                    if 'childclass' in body.attrib and body.get('childclass') == default.attrib['class']:
                        body.set('childclass', '{}_{}'.format(id, body.attrib['childclass']))
                default.set('class', '{}_{}'.format(id, default.attrib['class']))

        for joint in node.findall('.//joint[@name]'):
            target = joint.attrib['name']
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == target:
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])

        for connect in node.findall('.//connect[@name]'):
            for element in node.findall('.//'):
                for key, value in element.attrib.items():
                    if value == connect.attrib['name']:
                        element.attrib[key] = '{}_{}'.format(id, element.attrib[key])

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

    def add_node_from_xml(self, attached_node: str = 'worldbody', xml_path: str = None):
        """ Add node from xml file. The attached node is the parent node of the new node.
        :param attached_node: the parent node of the new node
        :param xml_path: the path of the xml file
        """
        if xml_path is None:
            raise ValueError("Please checkout your xml path.")
        if attached_node == 'worldbody':
            parent_element = self.root.find('worldbody')
        else:
            parent_element = self.root.find(f'.//body[@name=\'{attached_node}\']')
        new_tree = ET.parse(xml_path)
        new_node = new_tree.getroot().find('worldbody').find('body')
        parent_element.append(new_node)

    def set_node_attrib(self, node: str, attrib: dict):
        """ Set node attribute.
        :param node: node name
        :param attrib: attribute dict
        """
        node_element = self.root.find(f'.//body[@name=\'{node}\']')
        for key in attrib:
            node_element.set(key, attrib[key])

    def add_node_from_str(self, father_node: str, xml_text: str):
        parent_element = self.root.find(father_node)
        new_element = ET.fromstring(xml_text)
        parent_element.append(new_element)

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
            body_element = self.root.find('worldbody').find(node)
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

    def save_xml(self, output_path='../assets'):
        """ Save xml file with identity path"""
        self.tree.write(path.join(output_path, f"{self.xml_name}.xml"))

    def save_and_load_xml(self, output_path=path.join(path.dirname(__file__), '../assets')):
        """ Save xml file and get its path"""
        self.save_xml(output_path)
        return path.abspath(path.join(output_path, f"{self.xml_name}.xml"))

    def splice_robot(self, scene=None, chassis=None, manipulator=None, gripper=None, **kwargs):
        """

        :param scene: scene name
        :param chassis: chassis name
        :param manipulator: manipulator name
        :param gripper: gripper name
        :param kwargs:
        """
        if isinstance(scene, str):
            if scene.endswith('.xml'):
                scene_path = scene
            else:
                scene_path = path.join(SCENES_DIR_PATH, '{}.xml'.format(scene))
            self._init_scene(scene_path)
        else:
            raise ValueError("Must has scene.xml to generate the world.")

        if isinstance(chassis, str):
            chassis_path = path.join(CHASSISES_DIR_PATH, chassis, '{}.xml'.format(chassis))
            self.add_component_from_xml(chassis_path, goal_body=(0, 'worldbody'))

        if isinstance(manipulator, str):
            if manipulator.endswith('.xml'):
                manipulator_path = manipulator
            else:
                manipulator_path = path.join(MANIPULATORS_DIR_PATH, manipulator, '{}.xml'.format(manipulator))
            self.add_component_from_xml(manipulator_path,
                                        goal_body=(0, '0_mount_base_link') if chassis is not None else (0, 'worldbody'))
            if isinstance(gripper, str):
                for goal_body in enumerate(kwargs['g2m_body']):
                    gripper_path = path.join(GRIPPERS_DIR_PATH, gripper, '{}.xml'.format(gripper))
                    self.add_component_from_xml(gripper_path, goal_body=goal_body)
