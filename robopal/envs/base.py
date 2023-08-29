import abc

import mujoco
from robopal.utils.renderers import MjRenderer


class MujocoEnv(object):
    """ This environment is the base class.

     :param xml_path(str): Load xml file from xml_path to build the mujoco model.
     :param is_render(bool): Choose if use the renderer to render the scene or not.
     :param renderer(str): choose official renderer with "viewer",
            another renderer with "mujoco_viewer"
     :param control_freq(int): Upper-layer control frequency.
            Note that high frequency will cause high time-lag.
    """

    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=1000,
                 is_camera_used=False):

        self.robot = robot
        self.is_render = is_render
        self.control_freq = control_freq

        self.mj_model = self.robot.robot_model
        self.mj_data = self.robot.robot_data

        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = 0
        self.control_timestep = 0
        self.robot_dof = self.robot.jnt_num

        self.renderer = None
        if is_render:
            self.renderer = MjRenderer(self.mj_model, self.mj_data, renderer, is_camera_used)

        self._initialize_time()
        self._set_init_pose()

    def step(self, action):
        """ 
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        if self.renderer is not None and self.renderer.render_paused:
            self.cur_time += 1
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.preStep(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    @abc.abstractmethod
    def preStep(self, action):
        """ Writes your own codes between mujoco forward and step you want to control.

        :param action: input actions
        :return: None
        """
        raise NotImplementedError

    def reset(self):
        """ Reset the simulate environment, in order to execute next episode.

        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._set_init_pose()
        mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self, mode="human"):
        """ render mujoco """
        if self.is_render is True:
            self.renderer.render()

    def close(self):
        """ close the environment. """
        self.renderer.close()

    def _initialize_time(self):
        """ Initializes the time constants used for simulation.

        :param control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = 0.0005
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        if self.control_freq <= 0:
            raise ValueError("Control frequency {} is invalid".format(self.control_freq))
        self.control_timestep = 1.0 / self.control_freq

    def _set_init_pose(self):
        """ Set or reset init joint position when called env reset func.

        """
        for i in range(len(self.robot.arm)):
            for j in range(len(self.robot.arm[i].joint_index)):
                self.mj_data.joint(self.robot.arm[i].joint_index[j]).qpos = self.robot.arm[i].init_pose[j]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_body_id(self, name: str):
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
