import os
import logging
from typing import Union, List, Any, Dict
from copy import deepcopy
import inspect

import mujoco
import numpy as np

import robopal
from robopal.commons.renderers import MjRenderer
import robopal.envs
import robopal.robots
from robopal.robots.base import BaseRobot
import robopal.commons.transform as R

ROBOPAL_PATH = os.path.dirname(inspect.getfile(robopal))
MJCF_PATH = os.path.join(ROBOPAL_PATH, "assets/robot.xml")


def make(env_name: str, **kwargs) -> "MujocoEnv":
    """ Create an environment instance. """
    return robopal.envs.REGISTERED_ENVS[env_name](**kwargs)


class MujocoEnv:
    """ This environment is the base class.

        :param robot(str): Load xml file from xml_path to build the mujoco model
        :param render_mode(str): Choose if you use the renderer to render the scene or not
        :param camera_in_render(str): Choose the camera
        :param control_freq(int): Upper-layer control frequency
        Note that high frequency will cause high time-lag
        :param is_show_camera_in_cv(bool): Use camera or not
    """

    metadata = {
        "render_modes": [
            None,
            "human",
            "rgb_array",
            "depth",
            "unity",
        ],
    }

    def __init__(
        self,
        robot: Union[BaseRobot, str],
        *,
        control_freq: int = 200,
        render_mode: str = 'human',
        is_show_camera_in_cv: bool = False,
        is_render_camera_offscreen = False,
        camera_in_render: str = "frontview",
        camera_in_window = "free",
    ):
        if isinstance(robot, str):
            try:
                self.robot: BaseRobot = robopal.robots.REGISTERED_ROBOTS[robot]()
            except KeyError:
                logging.error(f"Robot {robot} is not registered. Available robots are {robopal.robots.REGISTERED_ROBOTS.keys()}.")
                raise KeyError
        else:
            self.robot = robot()
        assert isinstance(self.robot, BaseRobot), "Please select a robot config file."
        
        self.agents = self.robot.agents
        self.control_freq = control_freq

        self.mj_model: mujoco.MjModel = self.robot.robot_model
        self.mj_data: mujoco.MjData = self.robot.robot_data

        # time infos
        self.cur_timestep = 0
        self.model_timestep = 0
        self.control_timestep = 0
        self._n_substeps = 1

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = MjRenderer(
            self.mj_model, self.mj_data, 
            render_mode = self.render_mode,
            is_show_camera_in_cv = is_show_camera_in_cv, 
            is_render_camera_offscreen = is_render_camera_offscreen,
            camera_in_render = camera_in_render,
            camera_in_window = camera_in_window
        )
        self.is_render_camera_offscreen = is_render_camera_offscreen
        
        self._initialize_time()
        self._set_init_qpos()

        self._mj_state = None

    @property
    def name(self):
        return self.__class__.__name__

    def step(self, action: Union[np.ndarray, Dict[str, np.ndarray]] = None) -> Any:
        """ 
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        if self.renderer.render_paused:
            self.cur_timestep += 1
            mujoco.mj_step(self.mj_model, self.mj_data, self._n_substeps)

        if self.renderer.exit_flag:
            self.close()

    def forward(self):
        """ Forward the simulation. """
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(
        self,
        seed: Union[int, None] = None,
        options: Union[Dict[str, Any], None] = None,
    ):
        """ Reset the simulate environment, in order to execute next episode. """
        np.random.seed(seed)
        
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.reset_object()
        self._set_init_qpos()
        if self.robot.end is not None:
            for agent in self.agents:
                self.robot.end[agent].reset()
        self.forward()

        if isinstance(options, dict):
            if "disable_reset_render" in options and options["disable_reset_render"]:
                return

    def reset_object(self):
        """ Set pose of the object. """
        # override mjcf with reseted model
        # mujoco.mj_saveLastXML(self.robot.mjcf_generator.get_xml_path(), self.mj_model)
        pass

    def render(self, mode=None) -> Union[None, np.ndarray]:
        """ render one frame in mujoco """
        if self.render_mode in ["human", "rgb_array", "depth"]:
            return self.renderer.render(mode)

    def close(self):
        """ close the environment. """
        if self.renderer is not None:
            self.renderer.close()
            import os
            logging.info("Enviroment has closed!")
            os._exit(0)

    def _initialize_time(self):
        """ Initializes the time constants used for simulation.

        :param control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_timestep = 0
        self.model_timestep = self.mj_model.opt.timestep
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        if self.control_freq <= 0:
            raise ValueError("Control frequency {} is invalid".format(self.control_freq))
        self.control_timestep = 1.0 / self.control_freq

    def _set_init_qpos(self):
        """ Set or reset init joint position when called env reset func. """
        for agent in self.robot.agents:
            self.set_joint_qpos(self.robot.init_qpos[agent], agent)
        self.forward()

    def set_joint_qpos(self, qpos: np.ndarray, agent: str = 'agent0'):
        """ Set joint position. """
        assert qpos.shape[0] == self.robot.jnt_num
        for j, per_arm_joint_names in enumerate(self.robot.arm_joint_names[agent]):
            self.mj_data.joint(per_arm_joint_names).qpos = qpos[j]

    def set_actuator_ctrl(self, torque: np.ndarray, agent: str = 'agent0'):
        """ Set joint torque. """
        assert torque.shape[0] == self.robot.jnt_num
        for j, per_arm_actuator_names in enumerate(self.robot.arm_actuator_names[agent]):
            self.mj_data.actuator(per_arm_actuator_names).ctrl = torque[j]

    def set_object_pose(self, obj_joint_name: str = None, obj_pose: np.ndarray = None):
        """ Set pose of the object. """
        if isinstance(obj_joint_name, str):
            assert obj_pose.shape[0] == 7
            self.mj_data.joint(obj_joint_name).qpos = obj_pose

    def set_site_pos(self, site_name: str = None, site_pos: np.ndarray = None):
        """ Set pose of the object. """
        if isinstance(site_name, str):
            site_id = self.get_site_id(site_name)
            assert site_pos.shape[0] == 3
            self.mj_model.site_pos[site_id] = site_pos

    def get_body_id(self, name: str):
        """ Get body id from body name.

        :param name: body name
        :return: body id
        """
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def get_body_jacp(self, name):
        """ Query the position jacobian of a mujoco body using a name string.

        :param name: The name of a mujoco body
        :return: The jacp value of the mujoco body
        """
        bid = self.get_body_id(name)
        jacp = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        """ Query the rotation jacobian of a mujoco body using a name string.

        :param name: The name of a mujoco body
        :return: The jacr value of the mujoco body
        """
        bid = self.get_body_id(name)
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacBody(self.mj_model, self.mj_data, None, jacr, bid)
        return jacr

    def get_body_pos(self, name: str):
        """ Get body position from body name.

        :param name: body name
        :return: body position
        """
        return self.mj_data.body(name).xpos.copy()

    def get_body_quat(self, name: str):
        """ Get body quaternion from body name.

        :param name: body name
        :return: body quaternion
        """
        return self.mj_data.body(name).xquat.copy()

    def get_body_rotm(self, name: str):
        """ Get body rotation matrix from body name.

        :param name: body name
        :return: body rotation matrix
        """
        return self.mj_data.body(name).xmat.copy().reshape(3, 3)

    def get_body_xvelp(self, name: str) -> np.ndarray:
        """ Get body velocity from body name.

        :param name: body name
        :return: translational velocity of the body
        """
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.mj_data.qvel)
        return xvelp.copy()

    def get_body_xvelr(self, name: str) -> np.ndarray:
        """ Get body rotational velocity from body name.

        :param name: body name
        :return: rotational velocity of the body
        """
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.mj_data.qvel)
        return xvelr.copy()

    def get_camera_pos(self, name: str):
        """ Get camera position from camera name.

        :param name: camera name
        :return: camera position
        """
        return self.mj_data.cam(name).pos.copy()

    def get_site_id(self, name: str):
        """ Get site id from site name.

        :param name: site name
        :return: site id
        """
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)

    def get_site_jacp(self, name):
        """ Query the position jacobian of a mujoco site using a name string.

        :param name: The name of a mujoco site
        :return: The jacp value of the mujoco site
        """
        sid = self.get_site_id(name)
        jacp = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        """ Query the rotation jacobian of a mujoco site using a name string.

        :param name: The name of a mujoco site
        :return: The jacr value of the mujoco site
        """
        sid = self.get_site_id(name)
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, None, jacr, sid)
        return jacr

    def get_site_pos(self, name: str):
        """ Get body position from site name.

        :param name: site name
        :return: site position
        """
        return self.mj_data.site(name).xpos.copy()

    def get_site_xvelp(self, name: str) -> np.ndarray:
        """ Get site velocity from site name.

        :param name: site name
        :return: translational velocity of the site
        """
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.mj_data.qvel)
        return xvelp.copy()

    def get_site_xvelr(self, name: str) -> np.ndarray:
        """ Get site rotational velocity from site name.

        :param name: site name
        :return: rotational velocity of the site
        """
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.mj_data.qvel)
        return xvelr.copy()

    def get_site_quat(self, name: str):
        """ Get site quaternion from site name.

        :param name: site name
        :return: site quaternion
        """
        rotm = self.get_site_rotm(name)
        return R.mat_2_quat(rotm)

    def get_site_rotm(self, name: str):
        """ Get site rotation matrix from site name.

        :param name: site name
        :return: site rotation matrix
        """
        return self.mj_data.site(name).xmat.copy().reshape(3, 3)

    def get_geom_id(self, name: Union[str, List[str]]):
        """ Get geometry id from its name.

        :param name: geometry name
        :return: geometry id
        """
        if isinstance(name, str):
            return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        else:
            ids = []
            for geom in name:
                id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                ids.append(id)
            return ids

    def save_state(self):
        """ Save the state of the mujoco model. """
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.mj_model, spec)
        state = np.empty(size, np.float64)
        mujoco.mj_getState(self.mj_model, self.mj_data, state, spec)
        self._mj_state = state
    
    def get_state(self):
        """ Get the saved state """
        return deepcopy(self._mj_state)

    def load_state(self, state: np.ndarray = None):
        """ Load the state of the mujoco model. """
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        if state is None:
            mujoco.mj_setState(self.mj_model, self.mj_data, self._mj_state, spec)
        else:
            state = np.array(state.flatten(), np.float64)
            mujoco.mj_setState(self.mj_model, self.mj_data, state, spec)

    def load_model_from_string(self, xml_string):
        """ Override model from string.

        :param xml_string: xml string
        """
        self.robot.build_from_string(xml_string)
        self.mj_model = self.robot.robot_model
        self.mj_data = self.robot.robot_data
        self.renderer._init_renderer(self.mj_model, self.mj_data)
        for end in self.robot.end.values():
            end.robot_data = self.mj_data

    def is_contact(self, geom1: Union[str, List[str]], geom2: Union[str, List[str]], verbose=False) -> bool:
        """ Check if two geom or geom list is in contact.

        :param geom1: geom name/list
        :param geom2: geom name/list
        :return: True/False
        """
        if isinstance(geom1, str):
            geom1 = [geom1]
        if isinstance(geom2, str):
            geom2 = [geom2]

        if len(self.mj_data.contact) > 0:
            for i, geom_pair in enumerate(self.mj_data.contact.geom):
                if geom_pair[0] in geom1 and geom_pair[1] in geom2:
                    break
                if geom_pair[0] in geom2 and geom_pair[1] in geom1:
                    break
            if verbose:
                contact_info = self.mj_data.contact[i]
                name1 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact_info.geom1)
                name2 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact_info.geom2)
                dist = contact_info.dist
                logging.info(f"contact geom: {name1} and {name2}")
                logging.info(f"dist: {dist}")
            return True
        return False
