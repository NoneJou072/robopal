import cv2
import numpy as np
from robopal.envs.jnt_ctrl_env import JntCtrlEnv
import robopal.commons.transform as trans
import robopal.commons.cv_utils as cv


class CamCalibEnv(JntCtrlEnv):
    """ Camera calibration environment.
    In this case, we will show the detail process of hand-eye calibration.
    Press 'Enter' to take a picture.
    """
    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="viewer",
                 control_freq=200,
                 jnt_controller='JNTIMP',
                 is_interpolate=False,
                 enable_camera_viewer=True,
                 cam_mode='rgb',
                 camera_name=None
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            jnt_controller=jnt_controller,
            is_interpolate=is_interpolate,
            enable_camera_viewer=enable_camera_viewer,
            cam_mode=cam_mode,
            camera_name=camera_name,
        )
        # Set low damping for easily dragging the end.
        self.jnt_controller.set_jnt_params(
            b=6.0 * np.ones(7),
            k=100.0 * np.ones(7),
        )
        self.camera_intrinsic_matrix = cv.get_cam_intrinsic()
        self.distCoeffs = np.zeros(5)
        print(self.camera_intrinsic_matrix)

    def step(self, action=None):
        action = self.robot.arm_qpos
        return super().step(action)


if __name__ == "__main__":
    from robopal.robots.diana_med import DianaCalib

    env = CamCalibEnv(
        robot=DianaCalib(),
        is_render=True,
        control_freq=200,
        jnt_controller='JNTIMP',
        is_interpolate=False,
        renderer='viewer',
        enable_camera_viewer=True,
        cam_mode='rgb',
        camera_name='cam'
    )
    env.reset()
    for t in range(int(1e6)):
        env.step()
        print(env.kdl_solver.fk(env.robot.arm_qpos))
        if env.is_render:
            env.render()
