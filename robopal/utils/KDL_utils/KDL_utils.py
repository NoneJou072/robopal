import os
import numpy as np
import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from robopal.utils.KDL_utils import KDL_main
from robopal.utils.KDL_utils.Update_Jaco3 import Update_Jaco as J_quat


class KDL_utils:

    def __init__(self,
                 urdf_path=os.path.join(os.path.dirname(__file__), "../images/DianaMed/urdf/DianaMed.urdf")):
        # Build kdl chain
        urdf = URDF.from_xml_file(urdf_path)
        self.tree = KDL_main.kdl_tree_from_urdf_model(urdf)
        self.chain = self.tree.getChain("base_link", "link7")  # default for DianaMed

        # Initialize solvers
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain, eps=0.0, maxiter=200, eps_joints=0.0)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacdot_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # Params
        self.NbOfJnt = self.chain.getNrOfJoints()
        self.env_grav = kdl.Vector(0.0, 0.0, -9.81)  # default
        self._dyn_kdl = kdl.ChainDynParam(self.chain, self.env_grav)
        self.mass_kdl = kdl.JntSpaceInertiaMatrix(self.NbOfJnt)
        self.corio_kdl = kdl.JntArray(self.NbOfJnt)
        self.grav_kdl = kdl.JntArray(self.NbOfJnt)
        self.ee_frame = kdl.Frame()
        print("KDL init !")

    def resetChain(self, base, end):
        self.chain = self.tree.getChain(base, end)  # default for DianaMed

        # Initialize solvers
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain, eps=0.0, maxiter=200, eps_joints=0.0)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacdot_solver = kdl.ChainJntToJacDotSolver(self.chain)

        # Params
        self.NbOfJnt = self.chain.getNrOfJoints()
        self.env_grav = kdl.Vector(0.0, 0.0, -9.81)  # default
        self._dyn_kdl = kdl.ChainDynParam(self.chain, self.env_grav)
        self.mass_kdl = kdl.JntSpaceInertiaMatrix(self.NbOfJnt)
        self.corio_kdl = kdl.JntArray(self.NbOfJnt)
        self.grav_kdl = kdl.JntArray(self.NbOfJnt)
        self.ee_frame = kdl.Frame()
        print("reset KDL Chain!")

    def setJntArray(self, q):
        q_kdl = kdl.JntArray(self.NbOfJnt)
        for i in range(len(q)):
            # for i in range(7):
            q_kdl[i] = q[i]
        return q_kdl

    def setNumpyMat(self, mat):
        if isinstance(mat, kdl.Rotation):
            m = np.zeros((3, 3))
        else:
            m = np.zeros((mat.rows(), mat.columns()))
            # m = np.zeros((7, 7))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i, j] = mat[i, j]
        return m

    def setNumpyArray(self, q):
        return np.asarray([q[i] for i in range(q.rows())], dtype=np.float32)

    def getInertiaMat(self, q):
        _q = self.setJntArray(q)
        self._dyn_kdl.JntToMass(_q, self.mass_kdl)
        return self.setNumpyMat(self.mass_kdl)

    def getCoriolisMat(self, q, qdot):
        _q = self.setJntArray(q)
        _qdot = self.setJntArray(qdot)
        self._dyn_kdl.JntToCoriolis(_q, _qdot, self.corio_kdl)
        return self.corio_kdl

    def getGravityMat(self, q):
        _q = self.setJntArray(q)
        self._dyn_kdl.JntToGravity(_q, self.grav_kdl)
        return self.grav_kdl

    def getCompensation(self, q, qdot):
        comp = np.zeros(7)
        self._dyn_kdl.JntToCoriolis(q, qdot, self.corio_kdl)
        self._dyn_kdl.JntToGravity(q, self.grav_kdl)
        for i in range(self.NbOfJnt):
            comp[i] = self.corio_kdl[i] + self.grav_kdl[i]
        return comp

    def getJac(self, q=kdl.JntArray()):
        j_ = kdl.Jacobian(self.NbOfJnt)
        self.jac_solver.JntToJac(q, j_)
        return j_

    def getJac_pinv(self, q=kdl.JntArray()):
        j_ = self.getJac(q)
        jacobian = np.empty((6, 7))
        for row in range(6):
            for col in range(7):
                jacobian[row][col] = j_.getColumn(col)[row]
        jacobian_pinv = np.linalg.pinv(jacobian)
        return jacobian_pinv

    def getJacQuaternion(self, q):
        return J_quat(q)  # output's type is numpyArray

    def getEEtf(self, joint_states):
        ee_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_states, ee_frame, self.NbOfJnt)
        return ee_frame

    def getEeCurrentPose(self, q):
        """ Get current pose with input's joint state.

        :param q: Current joint states
        :return: Current position and rotation
        """
        self.fk_solver.JntToCart(self.setJntArray(q), self.ee_frame, self.NbOfJnt)
        pos = np.array([self.ee_frame.p[i] for i in range(3)])
        rot = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                rot[i][j] = self.ee_frame.M[i, j]
        return pos, rot

    def ikSolver(self, pos: np.ndarray, rot: np.ndarray, q_init: np.ndarray = None) -> np.ndarray:
        if q_init is None:
            raise NotImplementedError
        q_init = self.setJntArray(q_init)

        for i in range(3):
            self.ee_frame.p[i] = pos[i]
            for j in range(3):
                self.ee_frame.M[i, j] = rot[i][j]
        q_result = kdl.JntArray(7)
        self.ik_solver.CartToJnt(q_init, self.ee_frame, q_result)
        return self.setNumpyArray(q_result)

    def jacobian_dot(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """

        Args:
            q (np.ndarray): current joint position  
            v (np.ndarray): current joint velocity

        Returns:
            np.ndarray: jacobian_dot
        """
        input_q = self.setJntArray(q)
        input_qd = self.setJntArray(v)
        input_qav = kdl.JntArrayVel(input_q, input_qd)
        output = kdl.Jacobian(self.NbOfJnt)
        self.jacdot_solver.JntToJacDot(input_qav, output)
        return self.setNumpyMat(output)

    def orientation_error(self, desired: np.ndarray, current: np.ndarray) -> np.ndarray:
        """computer ori error from ori to cartesian 姿态矩阵的偏差3*3的
        Args:
            desired (np.ndarray): desired orientation
            current (np.ndarray): current orientation

        Returns:
            _type_: orientation error(from pose(3*3) to eulor angular(3*1))
        """
        rc1 = current[:, 0]
        rc2 = current[:, 1]
        rc3 = current[:, 2]
        rd1 = desired[:, 0]
        rd2 = desired[:, 1]
        rd3 = desired[:, 2]
        if (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3)).all() <= 0.0001:
            w1, w2, w3 = 0.5, 0.5, 0.5
        else:
            w1, w2, w3 = 0.9, 0.5, 0.3

        error = w1 * np.cross(rc1, rd1) + w2 * np.cross(rc2, rd2) + w3 * np.cross(rc3, rd3)

        return error
