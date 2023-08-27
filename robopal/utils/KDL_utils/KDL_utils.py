import numpy as np
from robopal.utils.KDL_utils.Update_Jaco3 import Update_Jaco as J_quat
import pinocchio as pin


class KDL_utils:
    def __init__(self, urdf_path: str):
        # Load the urdf model
        urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path)

        # Create data required by the algorithms
        self.data = self.model.createData()

        self.JOINT_NUM = self.model.nq
        print(f"pinocchio model {self.model.name} init!")

    def fk(self, q: np.ndarray):
        """ Perform the forward kinematics over the kinematic tree

        :param q: joint position
        :return: end's translation, rotation
        """
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[-1].translation, self.data.oMi[-1].rotation

    def ik(self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """ Position the end effector of a manipulator robot to a given pose (position and orientation)
            The method employs a simple Jacobian-based iterative algorithm, which is called closed-loop inverse kinematics (CLIK).

        :param pos:
        :param rot:
        :param q_init:
        :return:
        """
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
            J = self.getJac(q)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            i += 1

        return q.flatten()

    def getInertiaMat(self, q: np.ndarray) -> np.ndarray:
        """

        :param q:
        :return:
        """
        return pin.crba(self.model, self.data, q)

    def getCoriolisMat(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """

        :param q:
        :param qdot:
        :return:
        """
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot)

    def getGravityMat(self, q: np.ndarray) -> np.ndarray:
        """

        :param q:
        :return:
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    def getJac(self, q: np.ndarray) -> np.ndarray:
        """ Computing the Jacobian in the joint frame

        :param q:
        :return:
        """
        return pin.computeJointJacobian(self.model, self.data, q, self.JOINT_NUM)

    def getJac_pinv(self, q: np.ndarray) -> np.ndarray:
        """ Computing the Jacobian_pinv in the joint frame

        :param q:
        :return:
        """
        return np.linalg.pinv(self.getJac(q))

    def getJacQuaternion(self, q) -> np.ndarray:
        """

        :param q:
        :return:
        """
        return J_quat(q)

    def get_jac_dot(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """

        :param q:
        :param v:
        :return:
        """
        pin.computeAllTerms(self.model, self.data, q, v)
        return self.data.dJ
