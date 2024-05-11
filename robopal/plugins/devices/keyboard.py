import time

from pynput import keyboard
import numpy as np
import robopal.commons.transform as T

class KeyboardIO:
    def __init__(self) -> None:

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )

        self._pos_step = 0.001
        self._rot_step = 0.001
        self._is_ctrl_l_pressed = False
        self._is_shift_pressed = False
        self._end_pos_offset = np.array([0.0, 0.0, 0.0])
        self._end_rot_offset = np.eye(3)
        
        self.command_introduction()

        listener.start()

    def command_introduction(self):
        print("Press arrow keys to move the end effector along the x/y-axis.")
        print("Press ctrl + arrow keys to move the end effector along the z-axis.")
        print("Press shift + arrow keys to rotate the end effector along the x/y-axis.")
        print("Press ctrl + shift + arrow keys to rotate the end effector along the z-axis.")
        print("Press esc to exit.")

    def on_press(self, key):
        if key == keyboard.Key.up:
            if self._is_ctrl_l_pressed: 
                if self._is_shift_pressed: # ctrl + shift + up
                    self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, 0, 1])))
                else:   # ctrl + up
                    self._end_pos_offset[2] += self._pos_step
            elif self._is_shift_pressed:  # shift + up
                self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([1, 0, 0])))
            else:   # up
                self._end_pos_offset[0] += self._pos_step
        elif key == keyboard.Key.down:
            if self._is_ctrl_l_pressed:
                if self._is_shift_pressed: # ctrl + shift + down
                    self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, 0, -1])))
                else:  # ctrl + down
                    self._end_pos_offset[2] -= self._pos_step
            elif self._is_shift_pressed:  # shift + down
                self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([-1, 0, 0])))
            else:  # down
                self._end_pos_offset[0] -= self._pos_step
        elif key == keyboard.Key.left:
            if self._is_shift_pressed:  # shift + left
                self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, 1, 0])))
            else:  # left
                self._end_pos_offset[1] -= self._pos_step
        elif key == keyboard.Key.right:
            if self._is_shift_pressed:  # shift + right
                self._end_rot_offset = self._end_rot_offset.dot(T.euler_2_mat(self._rot_step * np.array([0, -1, 0])))
            else:  # right
                self._end_pos_offset[1] += self._pos_step

        elif key == keyboard.Key.ctrl_l:
            self._is_ctrl_l_pressed = True
        
        elif key == keyboard.Key.shift:
            self._is_shift_pressed = True

    def on_release(self, key):
        # reset end offset
        self._end_pos_offset = np.zeros(3)
        self._end_rot_offset = np.eye(3)

        if key == keyboard.Key.ctrl_l:
            self._is_ctrl_l_pressed = False

        elif key == keyboard.Key.shift:
            self._is_shift_pressed = False

        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def get_end_pos_offset(self):
        return self._end_pos_offset
    
    def get_end_rot_offset(self):
        return self._end_rot_offset
