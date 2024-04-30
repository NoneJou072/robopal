import logging

import numpy as np
import pinocchio as pin
import robopal.commons.transform as T


class PinSolver:
    """ Pinocchio solver for kinematics and dynamics """

    def __init__(self, urdf_path: str):
        # Load the urdf model
        urdf_path = urdf_path

        # Create data required by the algorithms
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self._JOINT_NUM = self.model.nq

        logging.info('Model name in Pinocchio: ' + self.model.name)
        logging.info('Dimension of the joint position: ' + str(self._JOINT_NUM))
        logging.info('Dimension of the joint velocity: ' + str(self.model.nv))
        logging.info(f"Pinocchio model has init.")

    def fk(self, q: np.ndarray, rot_format: str = 'matrix'):
        """ Perform the forward kinematics over the kinematic tree

        :param q: joint position
        :param rot_format: 'matrix' or 'quaternion'
        :return: end's translation, rotation
        """
        pin.forwardKinematics(self.model, self.data, q)
        if rot_format == 'matrix':
            return self.data.oMi[-1].translation.copy(), self.data.oMi[-1].rotation.copy()
        elif rot_format == 'quaternion':
            return self.data.oMi[-1].translation.copy(), T.mat_2_quat(self.data.oMi[-1].rotation.copy())

    def ik(self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """ Position the end effector of a manipulator robot to a given pose (position and orientation)
            The method employs a simple Jacobian-based iterative algorithm, which is called closed-loop inverse kinematics (CLIK).

        :param pos: desired position
        :param rot: desired quaternion
        :param q_init: initial joint position
        :return: joint position
        """
        rot = T.quat_2_mat(rot)
        oM_des = pin.SE3(rot, pos)
        q = q_init

        eps = 1e-4
        IT_MAX = 1000
        DT = 1e-1
        damp = 1e-12

        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[-1].actInv(oM_des)
            err = pin.log(iMd).vector  # in joint frame
            if np.linalg.norm(err) < eps:
                break
            if i >= IT_MAX:
                break
            J = self.get_joint_jac(q)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            i += 1

        return q.flatten()

    def get_inertia_mat(self, q: np.ndarray) -> np.ndarray:
        """ Computing the inertia matrix in the joint frame

        :param q: joint position
        :return: inertia matrix
        """
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """ Computing the Coriolis matrix in the joint frame

        :param q: joint position
        :param qdot: joint velocity
        :return:
        """
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q: np.ndarray) -> np.ndarray:
        """ Computing the gravity matrix in the joint frame

        :param q: joint position
        :return: gravity matrix
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    def get_full_jac2(self, q: np.ndarray) -> np.ndarray:
        """ Computes the full model Jacobian, expressed in the coordinate world frame.

        :param q: joint position
        :return: Jacobian
        """
        return pin.computeJointJacobians(self.model, self.data, q).copy()

    def get_full_jac(self, q: np.ndarray) -> np.ndarray:
        """ Computes the full model Jacobian, expressed in the coordinate world frame.

        :param q: joint position
        :return: Jacobian
        """
        END_BODY_NAME = self.model.frames[-1].name
        IDX_TOOL = self.model.getFrameId(END_BODY_NAME)
        return pin.computeFrameJacobian(self.model, self.data, q, IDX_TOOL, pin.LOCAL_WORLD_ALIGNED)

    def get_joint_jac(self, q: np.ndarray) -> np.ndarray:
        """ Computes the Jacobian of a specific joint frame expressed in the local frame.

        :param q: joint position
        :return: Jacobian
        """
        return pin.computeJointJacobian(self.model, self.data, q, self._JOINT_NUM).copy()

    def get_joint_jac_pinv(self, q: np.ndarray) -> np.ndarray:
        """ Computes the full model Jacobian_pinv of a specific joint frame expressed in the local frame.

        :param q: joint position
        :return: Jacobian_pinv
        """
        return np.linalg.pinv(self.get_joint_jac(q)).copy()

    def get_full_jac_pinv(self, q: np.ndarray) -> np.ndarray:
        """ Computes the full model Jacobian_pinv expressed in the coordinate world frame.

        :param q: joint position
        :return: Jacobian_pinv
        """
        return np.linalg.pinv(self.get_full_jac(q)).copy()

    def get_jac_dot(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """ Computing the Jacobian_dot in the joint frame

        :param q: joint position
        :param v: joint velocity
        :return: Jacobian_dot
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.computeAllTerms(self.model, self.data, q, v)
        return self.data.dJ.copy()

    def get_end_vel(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """ Computing the end effector velocity

        :param q: joint position
        :param qd: joint velocity
        :return: end effector velocity, 6*1, [v, w]
        """
        return np.dot(self.get_full_jac(q), qd)
