class BaseDevice(object):
    def __init__(self, pos_scale=0.01, rot_scale=0.01) -> None:

        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self._reset_flag = False
        self._exit_flag = False
        self._gripper_flag = 0
        self._agent_id = 0

    def start(self):
        pass

    def command_introduction(self):
        pass

    def get_outputs(self):
        pass
