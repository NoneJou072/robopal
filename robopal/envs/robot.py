from typing import Union, Dict

import numpy as np

from robopal.envs.base import MujocoEnv
from robopal.controllers import controllers, BaseController


class RobotEnv(MujocoEnv):
    """ Robot environment.

    :param robot: Robot configuration.
    :param render_mode: Choose the render mode.
    :param controller: Choose the controller.
    :param control_freq: Upper-layer control frequency. i.g. frame per second-fps
            Note that high frequency will cause high time-lag.
    :param is_interpolate: Use interpolator while stepping.
    :param enable_camera_viewer: Use camera or not.
    """

    def __init__(self,
                 robot=None,
                 control_freq=200,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 camera_name=None,
                 render_mode='human',
                 ):

        super().__init__(
            robot=robot,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            camera_name=camera_name,
            render_mode=render_mode,
        )
        self.name = "default_env"
        self.is_interpolate = is_interpolate

        # choose controller
        assert controller in controllers, f"Not supported controller, you can choose from {controllers.keys()}"
        self.controller: BaseController = controllers[controller](
            self.robot,
            is_interpolate=is_interpolate,
            interpolator_config={'dof': self.robot.jnt_num, 'control_timestep': self.control_timestep}
        )

        # check the control frequency
        self._n_substeps = int(self.control_timestep / self.model_timestep)
        if self._n_substeps == 0:
            raise ValueError("Control frequency is too low. Checkout you are not in renderer mode."
                             "Current Model-Timestep:{}".format(self.model_timestep))
        if self.robot.end is not None:
            for agent in self.agents:
                self.robot.end[agent].dt = self.dt

        # memorize the initial position and rotation
        self.init_pos = dict()
        self.init_quat = dict()
        for agent in self.agents:
            self.init_pos[agent], self.init_quat[agent] = self.controller.forward_kinematics(self.robot.get_arm_qpos(agent), agent)
        self.robot.init_pos = self.init_pos
        self.robot.init_quat = self.init_quat

    def auto_render(func):
        """ Automatically render the scene. """
        def wrapper(self, *args, **kwargs):
            ret = func(self, *args, **kwargs)
            self.render()
            return ret

        return wrapper 

    @auto_render
    def step(self, action: Union[np.ndarray, Dict[str, np.ndarray]]):
        if self.is_interpolate:
            self.controller.step_interpolator(action)

        joint_inputs = self.controller.step_controller(action)
        # Send joint_inputs to simulation
        if isinstance(joint_inputs, np.ndarray):
            self.set_actuator_ctrl(joint_inputs)
        else:
            for agent in self.agents:
                self.set_actuator_ctrl(joint_inputs[agent], agent)

        super().step()
    
    @auto_render
    def reset(self, seed=None, options=None):
        self.controller.reset()
        super().reset(seed, options)

    def get_configs(self):
        """ Get global configs of the current enviroment.
        """
        return {
            'name': self.name,
            'robot': self.robot.name,
            'control_freq': self.control_freq,
            'controller': self.controller.name,
        }

    @property
    def dt(self):
        """ Time of each upper step in the environment.
        """
        return self._n_substeps * self.mj_model.opt.timestep
