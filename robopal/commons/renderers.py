import time

import mujoco
from collections import deque
import sys
import numpy as np
from queue import Queue
# TODO: separate a script
try:
    import cv2
except:
    print('Could not import cv2, please install it to enable camera viewer.')


class MjRenderer:
    def __init__(self, mj_model, mj_data, is_render, renderer, enable_camera_viewer=False, cam_mode='rgb'):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.renderer = renderer
        self.enable_camera_viewer = enable_camera_viewer
        self.cam_mode = cam_mode

        # Set up mujoco viewer
        self.viewer = None
        if is_render:
            self._init_renderer()

        self.traj = deque(maxlen=200)

        # keyboard flag
        self.render_paused = True
        self.exit_flag = False

        self._image = None
        self.image_queue = Queue()

        self.image_renderer = mujoco.Renderer(self.mj_model)

    def _init_renderer(self):
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """

        def key_callback(keycode):
            if keycode == 32:
                self.render_paused = not self.render_paused
            elif keycode == 256:
                self.exit_flag = not self.exit_flag

        if self.renderer == "unity":
            # TODO: Support unity renderer.
            raise ValueError("Unity renderer not supported now.")
        elif self.renderer == "viewer":
            from mujoco import viewer
            # This function does not block, allowing user code to continue execution.
            self.viewer = viewer.launch_passive(self.mj_model, self.mj_data, key_callback=key_callback)

            if self.enable_camera_viewer:
                cv2.namedWindow('RGB Image', cv2.WINDOW_NORMAL)
        else:
            raise ValueError('Invalid renderer name.')

    def render(self):
        """ render mujoco """
        if self.viewer is not None and self.render_paused is True and self.renderer == "viewer":
            if self.viewer.is_running() and self.exit_flag is False:
                self.viewer.sync()
            else:
                self.close()

            if self.enable_camera_viewer:
                enable_depth = True if self.cam_mode == 'depth' else False
                rgb = self.render_pixels_from_camera('0_cam', enable_depth=enable_depth)
                cv2.imshow('RGB Image', rgb)
                cv2.waitKey(1)

    def close(self):
        """ close the environment. """
        if self.enable_camera_viewer:
            cv2.destroyAllWindows()
        self.viewer.close()
        sys.exit(0)

    def set_renderer_config(self):
        """ Setup mujoco global config while using viewer as renderer.
            It should be noted that the render thread need locked.
        """
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.mj_data.time % 2)

    def render_traj(self, pos: np.ndarray):
        """ Render the trajectory from deque above,
            you can push the cartesian position into this deque.

        :param pos: One of the cartesian position of the trajectory to render.
        """
        if self.renderer == "viewer":
            self.traj.append(pos.copy())
            self.viewer.user_scn.ngeom = len(self.traj)
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

    def get_cam_intrinsic(self, fovy=45.0, width=320, height=240):
        aspect = width * 1.0 / height
        fovx = np.degrees(2 * np.arctan(aspect * np.tan(np.radians(fovy / 2))))

        cx = 0.5 * width
        cy = 0.5 * height
        fx = cx / np.tan(fovx * np.pi / 180 * 0.5)
        fy = cy / np.tan(fovy * np.pi / 180 * 0.5)

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        return K

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
