import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig


class OTG:
    def __init__(self,
                 OTG_dim=7,
                 control_cycle=0.001,
                 max_velocity=0.0,
                 max_acceleration=0.0,
                 max_jerk=0.0):

        self.otg = Ruckig(OTG_dim, control_cycle)

        self.inp = InputParameter(OTG_dim)
        self.out = OutputParameter(OTG_dim)

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

    def update_target_position(self, action):
        self.inp.target_position = action

    def update_state(self):
        self.otg.update(self.inp, self.out)
        q_target = self.out.new_position
        qd_target = self.out.new_velocity
        self.out.pass_to_input(self.inp)
        return q_target, qd_target
