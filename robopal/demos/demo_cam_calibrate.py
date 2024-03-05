import numpy as np

from robopal.envs.robot import RobotEnv
import robopal.commons.cv_utils as cv
from robopal.robots.diana_med import DianaCalib


class CamCalibEnv(RobotEnv):
    """ Camera calibration environment.
    In this case, we will show the detail process of hand-eye calibration.
    Press 'Enter' to take a picture.
    """
    def __init__(self,
                 robot=None,
                 render_mode='human',
                 control_freq=200,
                 controller='JNTIMP',
                 is_interpolate=False,
                 enable_camera_viewer=True,
                 camera_name=None
                 ):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            controller=controller,
            is_interpolate=is_interpolate,
            enable_camera_viewer=enable_camera_viewer,
            camera_name=camera_name,
        )
        # Set low damping for easily dragging the end.
        self.controller.set_jnt_params(
            b=6.0 * np.ones(7),
            k=100.0 * np.ones(7),
        )
        self.camera_intrinsic_matrix = cv.get_cam_intrinsic()
        self.distCoeffs = np.zeros(5)
        print(self.camera_intrinsic_matrix)

    def step(self, action=None):
        action = self.robot.get_arm_qpos()
        return super().step(action)


if __name__ == "__main__":

    env = CamCalibEnv(
        robot=DianaCalib(),
        render_mode='human',
        control_freq=200,
        controller='JNTIMP',
        is_interpolate=False,
        enable_camera_viewer=True,
        camera_name='cam'
    )
    env.reset()
    for t in range(int(1e6)):
        env.step()
        print(env.kd_solver.fk(env.robot.get_arm_qpos()))
