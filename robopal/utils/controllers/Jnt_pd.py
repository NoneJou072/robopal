import numpy as np


class Jnt_PD(object):
    def __init__(self, robot):
        # hyper-parameters of PD
        self.kp = robot.kp
        self.kd = robot.kd

        print("Jnt_PD Initialized!")

    def set_jnt_params(self, m: np.ndarray, b: np.ndarray, k: np.ndarray):
        pass

    def compute_jnt_torque(
            self,
            q_des: np.ndarray,
            v_des: np.ndarray,
            q_cur: np.ndarray,
            v_cur: np.array,
            *args,
            **kwargs
    ):
        """ Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_target: joint position
        :param qdot_target: joint velocity
        """
        acc_desire = self.kp * (q_des - q_cur) + self.kd * (v_des - v_cur)

        tau_target = np.dot(kwargs['M'], acc_desire) + kwargs['coriolis_gravity']
        return tau_target
