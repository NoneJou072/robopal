import numpy as np
import mujoco
from mujoco import minimize
from robopal.envs.robot import RobotEnv
import robopal.commons.transform as T


class PosCtrlEnv(RobotEnv):
    def __init__(self,
                 robot=None,
                 control_freq=200,
                 enable_camera_viewer=False,
                 controller='JNTIMP',
                 is_interpolate=False,
                 is_pd=False,
                 render_mode='human',
                 ):
        super().__init__(
            robot=robot,
            control_freq=control_freq,
            enable_camera_viewer=enable_camera_viewer,
            controller=controller,
            is_interpolate=is_interpolate,
            render_mode=render_mode,
        )

        self.p_cart = 0.2
        self.d_cart = 0.01
        self.p_quat = 0.2
        self.d_quat = 0.01
        self.is_pd = is_pd
        self.vel_des = np.zeros(3)

    def compute_pd_increment(self, p_goal: np.ndarray,
                             p_cur: np.ndarray,
                             r_goal: np.ndarray,
                             r_cur: np.ndarray,
                             pd_goal: np.ndarray = np.zeros(3),
                             pd_cur: np.ndarray = np.zeros(3)):
        pos_incre = self.p_cart * (p_goal - p_cur) + self.d_cart * (pd_goal - pd_cur)
        quat_incre = self.p_quat * (r_goal - r_cur)
        return pos_incre, quat_incre

    def step_controller(self, action: np.ndarray, agent: str = 'arm0'):
        assert len(action) in (3, 7), "Invalid action length."

        if not self.is_pd:
            p_goal = action[:3]
            r_goal = self.init_quat[agent] if len(action) == 3 else action[3:]
        else:
            p_cur, r_cur = self.controller.forward_kinematics(self.robot.get_arm_qpos(agent))
            r_target = self.init_quat[agent] if len(action) == 3 else action[3:]
            pd_cur = self.kd_solver.get_end_vel(self.robot.get_arm_qpos(agent), self.robot.get_arm_qvel(agent))
            p_incre, r_incre = self.compute_pd_increment(p_goal=action[:3], p_cur=p_cur,
                                                         r_goal=r_target, r_cur=r_cur,
                                                         pd_goal=self.vel_des, pd_cur=pd_cur[:3])
            p_goal = p_incre + p_cur
            r_goal = r_cur + r_incre

        return self.ik(p_goal, r_goal, agent)

    def step(self, action):
        if self.robot.agent_num == 1:
            inputs = self.step_controller(action)
        else:
            inputs = {agent: self.step_controller(action[agent], agent) for agent in self.robot.agents}

        super().step(inputs)

    def ik(self, pos, quat, agent='arm0', q_init=None):
        del q_init

        x = self.robot.get_arm_qpos(agent)
        x_prev = x.copy()
        
        ik_target = lambda x: self._ik_res(x, pos=pos, quat=quat, reg_target=x_prev, radius=20, reg=.001, agent=agent)
        jac_target = lambda x, r: self._ik_jac(x, r, pos=pos, quat=quat, radius=20, reg=.01, agent=agent)
        x, _ = minimize.least_squares(x, ik_target, self.robot.mani_joint_bounds[agent], jacobian=jac_target, eps=1e-4, verbose=0)

        return x
    
    def _ik_res(self, x, pos=None, quat=None, radius=6, reg=1e-3, reg_target=None, agent='arm0'):
        """Residual for inverse kinematics.

        Args:
            x: joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The residual of the Inverse Kinematics task.
        """

        # Position residual.
        p_cur, r_cur = self.controller.forward_kinematics(x, agent)
        
        res_pos = p_cur - pos

        # Orientation residual: quaternion difference.
        res_quat = np.empty(3)
        mujoco.mju_subQuat(res_quat, quat, r_cur)
        res_quat *= radius

        # Regularization residual.
        reg_target = self.robot.init_qpos[agent] if reg_target is None else reg_target
        res_reg = reg * (x - reg_target)

        return np.hstack((res_pos, res_quat, res_reg))
    
    def _ik_jac(self, x, res, pos=None, quat=None, radius=.04, reg=1e-3, agent='arm0'):
        """Analytic Jacobian of inverse kinematics residual

        Args:
            x: joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The Jacobian of the Inverse Kinematics task.
        """
        # least_squares() passes the value of the residual at x which is sometimes
        # useful, but we don't need it here.
        del res

        # We can assume x has been copied into qpos
        # and that mj_kinematics has been called by ik()

        # Call mj_comPos (required for Jacobians).
        mujoco.mj_comPos(self.mj_model, self.robot.kine_data)

        # Get end-effector site Jacobian.
        jac_pos = np.empty((3, self.mj_model.nv))
        jac_quat = np.empty((3, self.mj_model.nv))

        mujoco.mj_jacBody(self.mj_model, self.robot.kine_data, jac_pos, jac_quat, self.robot.kine_data.body(self.robot.end_name[agent]).id)
        jac_pos = jac_pos[:, self.robot.arm_joint_indexes[agent]]
        jac_quat = jac_quat[:, self.robot.arm_joint_indexes[agent]]
        # Get Deffector, the 3x3 mju_subquat Jacobian
        effector_quat = np.empty(4)
        mujoco.mju_mat2Quat(effector_quat, self.robot.kine_data.body(self.robot.end_name[agent]).xmat)
        target_quat = quat
        Deffector = np.empty((3, 3))
        mujoco.mjd_subQuat(target_quat, effector_quat, None, Deffector)

        # Rotate into target frame, multiply by subQuat Jacobian, scale by radius.
        target_mat = T.quat_2_mat(quat)
        mat = radius * Deffector.T @ target_mat.T
        jac_quat = mat @ jac_quat

        # Regularization Jacobian.
        jac_reg = reg * np.eye(7)

        return np.vstack((jac_pos, jac_quat, jac_reg))
    