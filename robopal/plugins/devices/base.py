class BaseDevice(object):
    def __init__(self) -> None:

        self._pos_step = 0.001
        self._rot_step = 0.001
        self._is_ctrl_l_pressed = False
        self._is_shift_pressed = False
        self._end_pos_offset = None
        self._end_rot_offset = None

        self._reset_flag = False
        self._exit_flag = False
        self._gripper_flag = 0
        self._agent_id = 0

    def start(self):
        pass

    def command_introduction(self):
        pass

    def get_end_pos_offset(self):
        pass
    
    def get_end_rot_offset(self):
        pass
