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
    :param is_show_camera_in_cv: Use camera or not.
    """
    
    def __init__(self,
                 robot,
                 *,
                 controller='JNTIMP',
                 control_freq=200,
                 is_interpolate=False,
                 render_mode='human',
                 is_show_camera_in_cv=False,
                 is_render_camera_offscreen = False,
                 camera_in_render="frontview",
                 camera_in_window="free",
                 ):

        super().__init__(
            robot=robot,
            control_freq=control_freq,
            render_mode=render_mode,
            is_show_camera_in_cv=is_show_camera_in_cv,
            is_render_camera_offscreen = is_render_camera_offscreen,
            camera_in_render=camera_in_render,
            camera_in_window = camera_in_window,
        )

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
        
        self.update_init_pose_to_current()

    def update_init_pose_to_current(self):
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
        return super().reset(seed, options)

    def get_configs(self, key: str = None):
        """ Get global configs of the current enviroment.
        """
        mjcf_content = None
        with open(self.robot.mjcf_generator.get_xml_path(), 'r') as f:
            mjcf_content = f.read()
        env_args = {
            'env_name': self.name,
            'robot': self.robot.name,
            'control_freq': self.control_freq,
            'controller': self.controller.name,
            'rng': None,  # todo: add rng
            'model_file': mjcf_content
        }

        if isinstance(key, str):
            assert key in env_args.keys(), "Invalid key."
            return env_args[key]
        else:
            return env_args

    def get_end_abs_pos(self, agent):
        return self.controller.forward_kinematics(self.robot.get_arm_qpos(agent), agent=agent)[0]

    def get_end_abs_quat(self, agent):
        return self.controller.forward_kinematics(self.robot.get_arm_qpos(agent), agent=agent)[1]

    @property
    def dt(self):
        """ Time of each upper step in the environment.
        """
        return self._n_substeps * self.mj_model.opt.timestep
