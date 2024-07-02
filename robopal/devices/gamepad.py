import logging

try:
    import pygame
except ImportError:
    logging.warn("pygame is not installed. If you want to use the gamepad, Please install it by running 'pip install pygame'")

import numpy as np
import robopal.commons.transform as T
from robopal.devices import BaseDevice


class Gamepad(BaseDevice):
    def __init__(self) -> None:
        self.joystick = None

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

        pygame.init()
        pygame.joystick.init()

        # ckeck if the gamepad is connected
        if pygame.joystick.get_count() == 0:
            logging.error("Gamepad not found. Please connect the gamepad and try again.")
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info(f"Gamepad has found: {self.joystick.get_name()}")

    def command_introduction(self):
        logging.info("Move <LS> to move the end effector along the x/y-axis.")
        logging.info("Press <LT/RT> to move the end effector along the z-axis.")
        # logging.info("Press <SHIFT + ARROW> to rotate the end effector along the x/y-axis.")
        # logging.info("Press <CTRL + SHIFT + ARROW> to rotate the end effector along the z-axis.")
        logging.info("Press <RB> to open/close the gripper.")
        logging.info("Press <X> to switch the agent.")
        logging.info("Press <ESC> to exit.")

    def listen(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_id = event.axis
                axis_value = self.joystick.get_axis(axis_id)

            if event.type == pygame.JOYBUTTONDOWN:
                button_id = event.button

                if button_id == 2:
                    self._agent_id = 0 if self._agent_id else 1
                elif button_id == 3:
                    self._reset_flag = True
                elif button_id == 5:
                    self._gripper_flag = not self._gripper_flag
                elif button_id == 8:
                    # Stop listener
                    self._exit_flag = True
                    return False
                
                for i in range(self.joystick.get_numbuttons()):
                    button = self.joystick.get_button(i)

            # if event.type == pygame.JOYBUTTONUP:
            #     print(f"按键 {event.button} 释放")

            if event.type == pygame.JOYHATMOTION:
                for i in range(self.joystick.get_numhats()):
                    hat = self.joystick.get_hat(i)

    def get_axis(self):
        """ read the axis value of the gamepad
        """
        self.listen()
        return {f"axis_{i}": self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())}
    
    def get_button(self):
        """ read the button value of the gamepad
        """
        self.listen()
        return {f"button_{i}": self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())}
    
    def get_outputs(self):
        self.listen()

        axis_value = self.get_axis()
        button_value = self.get_button()

        pos_increment_scale = 0.01
        pos_increment = np.array([
            axis_value["axis_1"], 
            axis_value["axis_0"], 
            0.5 * (axis_value["axis_5"] - axis_value["axis_2"]) * (1 - button_value["button_10"])
        ]) * pos_increment_scale

        rot_increment_scale = 0.1
        rot_increment_x = axis_value["axis_3"]
        rot_increment_y = -axis_value["axis_4"]
        rot_increment_z = 0.5 * (axis_value["axis_5"] - axis_value["axis_2"]) * button_value["button_10"]
        rot_increment = np.array([rot_increment_x, rot_increment_y, rot_increment_z]) * rot_increment_scale

        return (
            np.clip(pos_increment, -0.04, 0.04),
            rot_increment
        )

    def __del__(self):
        pygame.joystick.quit()
        pygame.quit()
