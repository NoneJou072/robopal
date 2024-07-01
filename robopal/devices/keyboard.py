import time
import logging

try:
    from pynput import keyboard
except ImportError:
    raise("pynput is not installed. Please install it by running 'pip install pynput'")
import numpy as np
import robopal.commons.transform as T
from robopal.devices import BaseDevice


class Keyboard(BaseDevice):
    def __init__(self) -> None:

        self._pos_step = 0.01
        self._rot_step = 0.01
        self._is_ctrl_l_pressed = False
        self._is_shift_pressed = False
        self._end_pos_offset = np.array([0.0, 0.0, 0.0])
        self._end_rot_offset = np.eye(3)

        self._reset_flag = False
        self._exit_flag = False
        self._gripper_flag = 0
        self._agent_id = 0

    def start(self):
        
        self.command_introduction()

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()

    def command_introduction(self):
        logging.info("Press <ARROW> to move the end effector along the x/y-axis.")
        logging.info("Press <CTRL + ARROW> to move the end effector along the z-axis.")
        logging.info("Press <SHIFT + ARROW> to rotate the end effector along the x/y-axis.")
        logging.info("Press <CTRL + SHIFT + ARROW> to rotate the end effector along the z-axis.")
        logging.info("Press <CAPSLOCK> to open/close the gripper.")
        logging.info("Press <ALT> to switch the agent.")
        logging.info("Press <ESC> to exit.")

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                if self._is_ctrl_l_pressed: 
                    if self._is_shift_pressed: # ctrl + shift + up
                        self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, 0, 1])))
                    else:   # ctrl + up
                        self._end_pos_offset[2] += self._pos_step
                elif self._is_shift_pressed:  # shift + up
                    self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, -1, 0])))
                else:   # up
                    self._end_pos_offset[0] += self._pos_step
            elif key == keyboard.Key.down:
                if self._is_ctrl_l_pressed:
                    if self._is_shift_pressed: # ctrl + shift + down
                        self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, 0, -1])))
                    else:  # ctrl + down
                        self._end_pos_offset[2] -= self._pos_step
                elif self._is_shift_pressed:  # shift + down
                    self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, 1, 0])))
                else:  # down
                    self._end_pos_offset[0] -= self._pos_step
            elif key == keyboard.Key.left:
                if self._is_shift_pressed:  # shift + left
                    self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([1, 0, 0])))
                else:  # left
                    self._end_pos_offset[1] += self._pos_step
            elif key == keyboard.Key.right:
                if self._is_shift_pressed:  # shift + right
                    self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([-1, 0, 0])))
                else:  # right
                    self._end_pos_offset[1] -= self._pos_step

            elif key == keyboard.Key.ctrl_l:
                self._is_ctrl_l_pressed = True
            
            elif key == keyboard.Key.shift:
                self._is_shift_pressed = True
            
            elif key == keyboard.Key.caps_lock:
                self._gripper_flag = not self._gripper_flag

        except AttributeError:
            pass

    def on_release(self, key):
        try:
            # reset end offset
            self._end_pos_offset = np.zeros(3)
            self._end_rot_offset = np.eye(3)

            if key == keyboard.Key.ctrl_l:
                self._is_ctrl_l_pressed = False

            elif key == keyboard.Key.shift:
                self._is_shift_pressed = False

            elif key == keyboard.Key.esc:
                # Stop listener
                self._exit_flag = True
                return False
            
            elif key.char == "r":
                self._reset_flag = True
            
            elif key == keyboard.Key.alt:
                self._agent_id = 0 if self._agent_id else 1

        except AttributeError:
            pass

    def get_end_pos_offset(self):
        return np.clip(self._end_pos_offset, -0.04, 0.04)
    
    def get_end_rot_offset(self):
        return self._end_rot_offset
