import sys
from collections import deque
import mujoco
import numpy as np


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
                 control_freq=1000):

        self.robot = robot
        self.is_render = is_render
        self.renderer = renderer
        self.control_freq = control_freq

        self.mj_model = self.robot.robot_model
        self.mj_data = self.robot.robot_data
        self.viewer = None
        self.cur_time = 0
        self.timestep = 0
        self.model_timestep = 0
        self.control_timestep = 0
        self.robot_dof = self.robot.jnt_num
        self.traj = deque(maxlen=200)

        # keyboard flag
        self.render_paused = True
        self.exit_flag = False

        self._renderer_init()
        self._initialize_time()
        self._set_init_pose()

    def step(self, action):
        """ 
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        if self.render_paused:
            self.cur_time += 1
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.preStep(action)
            mujoco.mj_step(self.mj_model, self.mj_data)
        if self.exit_flag is True:
            self.close()

    def reset(self):
        """ Reset the simulate environment, in order to execute next episode.

        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._set_init_pose()
        mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self, mode="human"):
        """ render mujoco """
        if self.is_render is True and self.viewer is not None and self.render_paused is True:
            if self.renderer == "mujoco_viewer":
                if self.viewer.is_alive is True:
                    self.viewer.render()
                else:
                    sys.exit(0)
            elif self.renderer == "viewer":
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    self.viewer.close()
                    sys.exit(0)

    def close(self):
        """ close the environment. """
        self.viewer.close()
        sys.exit(0)

    def _renderer_init(self):
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """
        def key_callback(keycode):
            if keycode == 32:
                self.render_paused = not self.render_paused
            elif keycode == 256:
                self.exit_flag = not self.exit_flag

        if self.is_render is True:
            if self.renderer == "mujoco_viewer":
                import mujoco_viewer
                self.viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data)
            elif self.renderer == "viewer":
                from mujoco import viewer
                # This function does not block, allowing user code to continue execution.
                self.viewer = viewer.launch_passive(self.mj_model, self.mj_data, key_callback=key_callback)

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

    def render_traj(self, pos):
        """ Render the trajectory from deque above,
            you can push the cartesian position into this deque.

        :param pos: One of the cartesian position of the trajectory to render.
        """
        if self.renderer == "mujoco_viewer" and self.is_render is True:
            if self.cur_time % 10 == 0:
                self.traj.append(pos.copy())
            for point in self.traj:
                self.viewer.add_marker(pos=point, size=np.array([0.001, 0.001, 0.001]), rgba=np.array([0, 0, 1, 1]),
                                       type=mujoco.mjtGeom.mjGEOM_SPHERE)

    def preStep(self, action):
        """ Writes your own codes between mujoco forward and step you want to control.

        :param action: input actions
        :return: None
        """
        raise NotImplementedError

    def get_body_id(self, name: str):
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    def setup_mujoco_config(self):
        """ Setup mujoco global config while using viewer as renderer.
            It should be noted that the render thread need locked.
        """
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.mj_data.time % 2)
