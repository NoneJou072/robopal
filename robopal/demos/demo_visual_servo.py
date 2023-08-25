import PyKDL as kdl
import numpy as np
import mujoco as mj
import cv2
import threading
import math
from scipy.spatial.transform import Rotation as RR
from robopal.envs.joint_pd_env import SingleArmEnv


class Visual_servo(SingleArmEnv):
    def __init__(self,
                 robot=None,
                 is_render=False,
                 renderer="mujoco_viewer",
                 control_freq=200,
                 is_interpolate=False
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate
        )
        self.Detected_already = False
        self.image = None
        self.gain = 2

        self.cameraMatrix = self.fov2Intrinsic()
        self.distCoeffs = np.zeros(5)
        self.aruco_parameters = cv2.aruco.DetectorParameters()

    @staticmethod
    def fov2Intrinsic(fovy=45.0, width=320, height=240):
        aspect = width * 1.0 / height
        fovx = math.degrees(2 * math.atan(aspect * math.tan(math.radians(fovy / 2))))

        cx = width / 2.0
        cy = height / 2.0
        fx = cx / math.tan(fovx * math.pi / 180 * 0.5)
        fy = cy / math.tan(fovy * math.pi / 180 * 0.5)

        K = [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]]
        return np.array(K, dtype=float)

    def camera_viewer(self):
        renderer = mj.Renderer(self.mj_model)
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

    def aruco_detection(self, marker_size, aruco_dict):
        try:
            if aruco_dict != "DICT_4X4_100":
                raise ValueError("Invalid aruco dictionary ID: " + aruco_dict)
        except ValueError as e:
            print(str(e))
        dictionary_id = cv2.aruco.__getattribute__(aruco_dict)
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

        cv_image = self.image
        R_vec = np.array([0.0, 0.0, 0.0])
        T_vec = np.array([0.0, 0.0, 0.0])

        if cv_image is not None:
            corners, marker_ids, _ = cv2.aruco.detectMarkers(cv_image,
                                                             aruco_dictionary, self.cameraMatrix, self.distCoeffs,
                                                             parameters=self.aruco_parameters)
            if marker_ids is not None:
                self.Detected_already = True
                if cv2.__version__ > '4.0.0':
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                          marker_size, self.cameraMatrix,
                                                                          self.distCoeffs)
                else:
                    rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                       marker_size, self.cameraMatrix,
                                                                       self.distCoeffs)

                # aruc_dec = np.array([tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2], 1, 0, 0,  0])
                T_vec = np.array([tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2]])
                R_vec = np.array([rvecs[0][0][0], rvecs[0][0][1], rvecs[0][0][2]])
            else:
                self.Detected_already = False
        return R_vec, T_vec

    def step(self, action):
        var_lambda = 1.6
        marker_size = 0.05
        aruco_dict = "DICT_4X4_100"
        gain = 0.01

        desire_p = np.array([0.0, 0., 0.2]).reshape(3, 1)
        desire_r = np.array([0.70575314, -0.70673064, 0.0, 0.0])

        cam_to_aruco_r, cam_to_aruco_p = self.aruco_detection(marker_size, aruco_dict)

        joints = self.kdl_solver.setJntArray(self.robot.single_arm.arm_qpos)
        jac_pinv = self.kdl_solver.getJac_pinv(joints)
        kdl_base2hand_f = self.kdl_solver.getEEtf(joints)

        hand_to_cam_p = np.array([0.0, 0.00, 0.007])
        hand_to_cam_r = np.array([-3.14, 0, 1.57])
        hand_to_cam_M = RR.from_euler('xyz', hand_to_cam_r).as_matrix()
        hand2cam_M = kdl.Rotation.RPY(hand_to_cam_r[0], hand_to_cam_r[1], hand_to_cam_r[2])
        hand2cam_v = kdl.Vector(hand_to_cam_p[0], hand_to_cam_p[1], hand_to_cam_p[2])
        kdl_hand2cam_f = kdl.Frame(hand2cam_M, hand2cam_v)
        kdl_base2cam_f = kdl_base2hand_f * kdl_hand2cam_f
        np_base2cam = self.kdl_solver.setNumpyMat(kdl_base2cam_f.M)

        if self.Detected_already:

            T_ca = cam_to_aruco_p.reshape(3, 1)
            R_ca = RR.from_rotvec(cam_to_aruco_r).as_matrix()

            T_acd = desire_p
            R_acd = RR.from_quat(desire_r).as_matrix()
            R_ccd = np.matmul(R_ca, R_acd)

            T_ccd_ = np.matmul(R_ca, T_acd)
            T_ccd = T_ccd_ + T_ca

            R_cd_c = np.transpose(R_ccd)
            R_cd_c_q = RR.from_matrix(R_cd_c).as_quat()
            T_cd_c = -np.matmul(R_cd_c, T_ccd)
            thu = RR.from_matrix(R_cd_c).as_rotvec()
            Error_R = thu.reshape(3, 1)

            error = np.concatenate([T_cd_c, Error_R], axis=0)
            print(np.linalg.norm(error))
            if np.linalg.norm(error) > gain:
                Lin_v = -var_lambda * np.matmul(R_ccd, T_cd_c)
                Ang_v = -var_lambda * Error_R

                V_camera = np.concatenate([Lin_v, Ang_v], axis=0)
                V_camera = V_camera.reshape(6, 1)

                Rv = np.vstack((np.hstack((np_base2cam, np.zeros((3, 3)))),
                                np.hstack((np.zeros((3, 3)), np_base2cam))))
                V_camera = np.matmul(Rv, V_camera)

                V_camera[1] = -V_camera[1]
                V_camera[2] = -V_camera[2]

                V_camera[4] = -V_camera[4]
                V_camera[5] = -V_camera[5]
            else:
                V_camera = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            V_camera = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        V_dian = np.dot(jac_pinv, V_camera)

        for i in range(7):
            action[i] = self.gain * V_dian[i] * 0.02 + joints[i]

        return super().step(action)


if __name__ == "__main__":
    from robopal.assets.robots.visual_servo import DianaMed

    env = Visual_servo(
        robot=DianaMed(),
        is_render=True,
        control_freq=200,
        is_interpolate=True,
        renderer='mujoco_viewer',
    )
    env.reset()
    env.cam_start()
    for t in range(int(1e6)):
        action = np.array([1, 5.75e-02, 1, 2.20e+00, -2.51e-02, 5.75e-01, 2.01e-02])
        env.step(action)
        if env.is_render:
            env.render()
