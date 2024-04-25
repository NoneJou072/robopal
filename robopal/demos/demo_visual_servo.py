import cv2
import numpy as np

from robopal.envs.robot import RobotEnv
import robopal.commons.transform as trans
import robopal.commons.cv_utils as cv


class VisualServo(RobotEnv):
    def __init__(self,
                 robot=None,
                 render_mode='human',
                 control_freq=200,
                 controller='JNTIMP',
                 is_interpolate=False,
                 enable_camera_viewer=True,
                 camera_name='0_cam'
                 ):
        super().__init__(
            robot=robot,
            control_freq=control_freq,
            controller=controller,
            is_interpolate=is_interpolate,
            enable_camera_viewer=enable_camera_viewer,
            camera_name=camera_name,
            render_mode=render_mode,
        )
        self.camera_name = camera_name
        self.camera_intrinsic_matrix = cv.get_cam_intrinsic()

        self.distCoeffs = np.zeros(5)
        aruco_parameters = cv2.aruco.DetectorParameters()
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)

    def aruco_detection(self, marker_size):
        cv_image = self.renderer.render_pixels_from_camera(cam=self.camera_name, enable_depth=False)
        if cv_image is not None:
            corners, marker_ids, _ = self.detector.detectMarkers(cv_image)
            if marker_ids is not None:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, self.camera_intrinsic_matrix, self.distCoeffs
                )
                return rvec.flatten()[:3], tvec.flatten()[:3], True
        return np.zeros(3), np.zeros(3), False

    def step(self, action=None):
        var_lambda = 1.6
        marker_size = 0.05
        gain = 0.2

        desire_p = np.array([0.0, 0., 0.2]).reshape(3, 1)
        desire_r = np.array([0.0, 0.70575314, -0.70673064, 0.0])

        hand2cam_p = np.array([0.0, 0.00, 0.007])
        hand2cam_r = np.array([-3.14, 0, 1.57])

        hand2cam_M = trans.euler_2_mat(hand2cam_r)

        kdl_hand2cam_f = trans.make_transform(hand2cam_p, hand2cam_M)

        base2hand_p, base2hand_r = self.kd_solver.fk(self.robot.get_arm_qpos())
        kdl_base2hand_f = trans.make_transform(base2hand_p, base2hand_r)

        kdl_base2cam_f = kdl_base2hand_f @ kdl_hand2cam_f
        np_base2cam = kdl_base2cam_f[:3, :3]

        cam2aruco_r, cam2aruco_p, detected_already = self.aruco_detection(marker_size)
        if detected_already:
            T_ca = cam2aruco_p.reshape(3, 1)
            R_ca = trans.vec2_mat(cam2aruco_r)

            T_acd = desire_p
            R_acd = trans.quat_2_mat(desire_r)

            R_c_cd = R_ca @ R_acd
            T_c_cd = R_ca @ T_acd + T_ca

            R_cd_c = R_c_cd.T
            T_cd_c = -np.matmul(R_cd_c, T_c_cd)
            error_R = trans.mat_2_vec(R_cd_c).reshape(3, 1)

            error = np.concatenate([T_cd_c, error_R], axis=0)

            if np.linalg.norm(error) > 0.01:
                Lin_v = -var_lambda * np.matmul(R_c_cd, T_cd_c)
                Ang_v = -var_lambda * error_R

                V_camera = np.concatenate([Lin_v, Ang_v], axis=0).reshape(6, 1)

                Rv = np.vstack((np.hstack((np_base2cam, np.zeros((3, 3)))),
                                np.hstack((np.zeros((3, 3)), np_base2cam))))
                V_camera = np.matmul(Rv, V_camera)

                V_camera[1] = -V_camera[1]
                V_camera[2] = -V_camera[2]
                V_camera[4] = -V_camera[4]
                V_camera[5] = -V_camera[5]
            else:
                V_camera = np.zeros(6)
        else:
            V_camera = np.zeros(6)

        jac_pinv = self.kd_solver.get_full_jac_pinv(self.robot.get_arm_qpos())
        V_dian = np.dot(jac_pinv, V_camera).reshape(-1)
        action = gain * V_dian + self.robot.get_arm_qpos()

        return super().step(action)


if __name__ == "__main__":
    from robopal.robots.diana_med import DianaAruco

    env = VisualServo(
        robot=DianaAruco(),
        render_mode='human',
        control_freq=100,
        controller='JNTIMP',
        enable_camera_viewer=True,
        camera_name='0_cam'
    )
    env.reset()
    for t in range(int(1e6)):
        env.step()
