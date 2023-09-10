from ruckig import InputParameter, OutputParameter, Result, Ruckig
import numpy as np


class OTG:
    def __init__(self,
                 OTG_Dof=7,
                 control_cycle=0.001,
                 max_velocity=0.0,
                 max_acceleration=0.0,
                 max_jerk=0.0):

        self.OTG_Dof = OTG_Dof
        self.control_cycle = control_cycle
        self.otg = Ruckig(self.OTG_Dof, self.control_cycle)

        self.inp = InputParameter(self.OTG_Dof)
        self.out = OutputParameter(self.OTG_Dof)

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def set_params(self, qpos, qvel):
        self.inp.current_position = qpos
        self.inp.current_velocity = qvel
        self.inp.current_acceleration = np.zeros(7)

        self.inp.target_position = np.zeros(7)
        self.inp.target_velocity = np.zeros(7)
        self.inp.target_acceleration = np.zeros(7)

        self.inp.max_velocity = self.max_velocity * np.ones(7)
        self.inp.max_acceleration = self.max_acceleration * np.ones(7)
        self.inp.max_jerk = self.max_jerk * np.ones(7)

    def update_input(self, action):
        self.inp.target_position = action

    def update_state(self):
        self.otg.update(self.inp, self.out)
        q_target = self.out.new_position
        qd_target = self.out.new_velocity
        self.out.pass_to_input(self.inp)
        return q_target, qd_target
