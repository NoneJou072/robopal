import logging
from queue import Queue
from collections import deque
from typing import Union, List

import numpy as np
import mujoco
from mujoco import viewer

import robopal.commons.cv_utils as cv

logging.basicConfig(level=logging.INFO)


class MjRenderer:
    def __init__(
        self,
        mj_model,
        mj_data,
        render_mode: Union[str, None] = 'human',
        is_show_camera_in_cv = False,
        is_render_camera_offscreen = False,
        camera_in_render = 'frontview',
        camera_in_window = "free",
    ):

        self.mj_model = mj_model
        self.mj_data = mj_data

        self.render_mode = render_mode
        self.is_render_camera_offscreen = is_render_camera_offscreen
        if is_show_camera_in_cv:
            assert cv.CV_FLAG, "OpenCV is not installed."
            assert self.is_render_camera_offscreen, "Camera should be rendered offscreen, please set is_render_camera_offscreen to True."
        self.is_show_camera_in_cv = is_show_camera_in_cv

        if self.render_mode in ["rgb_array", "depth"]:
            assert self.is_render_camera_offscreen, "Camera should be rendered offscreen, please set is_render_camera_offscreen to True."

        self.camera_in_render = camera_in_render
        self.camera_in_window = camera_in_window

        # keyboard flag
        self.enable_viewer_keyboard = True  # enable keyboard control in viewer
        self.render_paused = True
        self.exit_flag = False

        # Set up sync mujoco viewer
        self.viewer = None
        if self.render_mode in ["human", "rgb_array", "depth"]:
            self._init_renderer(mj_model, mj_data)
        elif self.render_mode is None:
            pass
        else:
            raise ValueError(f'{self.render_mode} is not a valid mode.')

        # image renderer
        if self.is_render_camera_offscreen:
            self.image_renderer = mujoco.Renderer(self.mj_model)
            self.image_queue = Queue(3)

        self.traj = deque(maxlen=200)  # used for rendering trajectory

    def key_callback(self, keycode):
        if self.enable_viewer_keyboard:
            if keycode == 32:  # space
                self.render_paused = not self.render_paused
            elif keycode == 256:  # esc
                self.exit_flag = True
            elif keycode == 257 and self.is_show_camera_in_cv :  # enter
                image = self.image_queue.get()
                cv.save_image(image)
                logging.info(f"Save a picture to {cv.CV_CACHE_DIR}.")

    def _init_renderer(self, mj_model, mj_data):
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """
        # refresh the data
        self.mj_model = mj_model
        self.mj_data = mj_data

        # set up the renderer
        if self.render_mode == "unity":
            # TODO: Support unity renderer.
            raise ValueError("Unity renderer not supported now.")
        elif self.render_mode in ["human", "rgb_array", "depth"]:
            # This function does not block, allowing user code to continue execution.
            if isinstance(self.viewer, viewer.Handle):
                self.viewer.close()
            self.viewer = viewer.launch_passive(mj_model, mj_data,
                                                key_callback=self.key_callback, 
                                                show_left_ui=False, show_right_ui=True)
            self.select_camera_view(self.camera_in_window)
            if self.is_show_camera_in_cv:
                cv.init_cv_window()
        else:
            raise ValueError('Invalid renderer name.')

    def render(self, mode: str = None):
        """ render per frame in glfw.
        """
        if self.render_paused and self.render_mode in ["human", "rgb_array", "depth"]:
            if isinstance(self.viewer, viewer.Handle):
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    self.close()

            if self.is_render_camera_offscreen:
                enable_depth = True if self.render_mode == 'depth' else False
                image = self.render_pixels_from_camera(self.camera_in_render, enable_depth=enable_depth)
                self.image_queue.put(image)
                if self.image_queue.full():
                    self.image_queue.get()

            if self.is_show_camera_in_cv:
                cv.show_image(image)
            
            if self.render_mode in ["rgb_array", "depth"] or mode == "rgb_array" or mode == "depth":
                assert self.is_render_camera_offscreen, "Camera should be rendered offscreen, please set is_render_camera_offscreen to True."
                return image
        return

    def close(self):
        """ close the environment. """
        if self.is_show_camera_in_cv :
            cv.close_cv_window()
        if isinstance(self.viewer, viewer.Handle) and self.viewer.is_running():
            self.viewer.close()
            del self.viewer
            logging.info("Viewer has closed!")
    
    def close_render_window(self):
        self.viewer.close()

    def select_camera_view(self, cam = "free"):
        """ Setup mujoco global config while using viewer as renderer.
            It should be noted that the render thread need locked.
        """
        with self.viewer.lock():
            if cam == "free":
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat = np.array([0.4, 0, 0.5])
            else:
                cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
                if cam_id >= 0:
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = cam_id
                else:
                    logging.warning(f"Camera {cam} not found.")
            self.viewer.cam.azimuth += 0.005

    def add_visual_point(self, pos: Union[np.ndarray, List[np.ndarray]]):
        """ Render the trajectory from deque above,
            you can push the cartesian position into this deque.

        :param pos: One of the cartesian position of the trajectory to render.
        """
        assert self.render_mode in ["human", "rgb_array", "depth"]
        if isinstance(pos, np.ndarray):
            self.traj.append(pos.copy())
            self.viewer.user_scn.ngeom = len(self.traj)
        else:
            self.traj = deque(maxlen=len(pos))
            for p in pos:
                self.traj.append(p.copy())
            self.viewer.user_scn.ngeom = len(pos)
        for i, point in enumerate(self.traj):
            # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.concatenate([np.random.uniform(0, 1, 3), np.array([1])], axis=0)
            )

    def visualize_site_frame(self):
        """ Visualize frames and labels. """
        assert self.render_mode in ["human", "rgb_array", "depth"]
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    
    def render_pixels_from_camera(self, cam, enable_depth=True):
        self.image_renderer.update_scene(self.mj_data, camera=cam)
        if enable_depth is True:
            self.image_renderer.enable_depth_rendering()
            org = self.image_renderer.render()
            image = org[:, :]
        else:
            org = self.image_renderer.render()
            image = org[:, :, ::-1]
        return image
