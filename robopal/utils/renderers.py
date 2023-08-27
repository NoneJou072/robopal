import mujoco
from collections import deque
import sys
import threading
import numpy as np


class MjRenderer:
    def __init__(self, mj_model, mj_data, renderer, is_camera_used):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.renderer = renderer
        self.viewer = None
        self.traj = deque(maxlen=200)

        # keyboard flag
        self.render_paused = True
        self.exit_flag = False

        self._renderer_init()

    def _renderer_init(self):
        """ Initialize renderer, choose official renderer with "viewer"(joined from version 2.3.3),
            another renderer with "mujoco_viewer"
        """

        def key_callback(keycode):
            if keycode == 32:
                self.render_paused = not self.render_paused
            elif keycode == 256:
                self.exit_flag = not self.exit_flag

        if self.renderer == "mujoco_viewer":
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.mj_model, self.mj_data)
        elif self.renderer == "viewer":
            from mujoco import viewer
            # This function does not block, allowing user code to continue execution.
            self.viewer = viewer.launch_passive(self.mj_model, self.mj_data, key_callback=key_callback)

    def render(self):
        """ render mujoco """
        if self.viewer is not None and self.render_paused is True:
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
                if self.exit_flag is True:
                    self.close()

    def close(self):
        """ close the environment. """
        self.viewer.close()
        sys.exit(0)

    def set_renderer_config(self):
        """ Setup mujoco global config while using viewer as renderer.
            It should be noted that the render thread need locked.
        """
        with self.viewer.lock():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.mj_data.time % 2)

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

    def camera_viewer(self):
        renderer = mujoco.Renderer(self.mj_model)
        while self.viewer.is_alive is True:
            renderer.update_scene(self.mj_data, camera="0_cam")
            org = renderer.render()
            self.image = org[:, :, ::-1]
            cv2.imshow('RGB Image', self.image)
            cv2.waitKey(1)

    def cam_start(self):
        cam_thread = threading.Thread(target=self.camera_viewer)
        cam_thread.daemon = True
        cam_thread.start()
