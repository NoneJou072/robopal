import logging
import time
import mujoco
from mujoco import viewer
from collections import deque
import sys
import numpy as np
from queue import Queue

import robopal.commons.cv_utils as cv

logging.basicConfig(level=logging.INFO)


class MjRenderer:
    def __init__(self, mj_model, mj_data, is_render, renderer,
                 enable_camera_viewer=False, cam_mode='rgb', camera_name='0_cam'):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.renderer = renderer
        self.enable_camera_viewer = enable_camera_viewer
        if cv.CV_FLAG is False:
            self.enable_camera_viewer = False
        self.cam_mode = cam_mode
        self.camera_name = camera_name

        # Set up mujoco viewer
        self.viewer = None
        if is_render:
            self._init_renderer()
            self.traj = deque(maxlen=200)  # used for rendering trajectory

        # keyboard flag
        self.render_paused = True
        self.exit_flag = False

        self._image = None
        self.image_queue = Queue(3)

        self.image_renderer = mujoco.Renderer(self.mj_model)

    def key_callback(self, keycode):
        if keycode == 32:  # space
            self.render_paused = not self.render_paused
        elif keycode == 256:  # esc
            self.exit_flag = not self.exit_flag
        elif keycode == 257:  # enter
            image = self.image_queue.get()
            cv.save_image(image)
            logging.info(f"Save a picture to {cv.CV_CACHE_DIR}.")
        if keycode == 265:  # Up arrow
            self.mj_data.mocap_pos[0, 2] += 0.01
        elif keycode == 264:  # Down arrow
            self.mj_data.mocap_pos[0, 2] -= 0.01
        elif keycode == 263:  # Left arrow
            self.mj_data.mocap_pos[0, 0] -= 0.01
        elif keycode == 262:  # Right arrow
            self.mj_data.mocap_pos[0, 0] += 0.01

    def _init_renderer(self):
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """
        if self.renderer == "unity":
            # TODO: Support unity renderer.
            raise ValueError("Unity renderer not supported now.")
        elif self.renderer == "viewer":
            # This function does not block, allowing user code to continue execution.
            self.viewer = viewer.launch_passive(self.mj_model, self.mj_data,
                                                key_callback=self.key_callback, show_left_ui=False, show_right_ui=True)
            self.set_renderer_config()
            if self.enable_camera_viewer:
                cv.init_cv_window()
        else:
            raise ValueError('Invalid renderer name.')

    def render(self):
        """ render mujoco """
        if self.viewer is not None and self.render_paused is True and self.renderer == "viewer":
            if self.viewer.is_running() and self.exit_flag is False:
                self.viewer: viewer.Handle
                self.viewer.sync()
            else:
                self.close()

            if self.enable_camera_viewer:
                enable_depth = True if self.cam_mode == 'depth' else False
                image = self.render_pixels_from_camera(self.camera_name, enable_depth=enable_depth)
                self.image_queue.put(image)
                if self.image_queue.full():
                    self.image_queue.get()
                cv.show_image(image)

    def close(self):
        """ close the environment. """
        if self.enable_camera_viewer:
            cv.close_cv_window()
        if self.viewer is not None:
            self.viewer.close()
        sys.exit(0)

    def set_renderer_config(self):
        """ Setup mujoco global config while using viewer as renderer.
            It should be noted that the render thread need locked.
        """
        self.viewer.cam.lookat = np.array([0.4, 0, 0.5])
        self.viewer.cam.azimuth -= 0.005
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.mj_data.time % 2)

    def add_visual_point(self, pos: np.ndarray | list[np.ndarray]):
        """ Render the trajectory from deque above,
            you can push the cartesian position into this deque.

        :param pos: One of the cartesian position of the trajectory to render.
        """
        assert self.renderer == "viewer"
        if isinstance(pos, np.ndarray):
            self.traj.append(pos.copy())
            self.viewer.user_scn.ngeom = len(self.traj)
        else:
            for p in pos:
                self.traj.append(p.copy())
            self.viewer.user_scn.ngeom = len(pos)
        for i, point in enumerate(self.traj):
            # Add a 3x3x3 grid of variously colored spheres to the middle of the scene.
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.concatenate([np.random.uniform(0, 1, 3), np.array([1])], axis=0)
            )

    def render_pixels_from_camera(self, cam='0_cam', enable_depth=True):
        self.image_renderer.update_scene(self.mj_data, camera=cam)
        if enable_depth is True:
            self.image_renderer.enable_depth_rendering()
            org = self.image_renderer.render()
            image = org[:, :]
        else:
            org = self.image_renderer.render()
            image = org[:, :, ::-1]
        self._image = image
        return image
