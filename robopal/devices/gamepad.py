import logging

try:
    import pygame
except ImportError:
    logging.warn("pygame is not installed. If you want to use the gamepad, Please install it by running 'pip install pygame'")

import numpy as np
import robopal.commons.transform as T
from robopal.devices import BaseDevice


class Gamepad(BaseDevice):
    def __init__(self, pos_scale=0.01, rot_scale=0.01) -> None:
        super().__init__(pos_scale, rot_scale)

        self.joystick = None

    def start(self):
        self.command_introduction()

        pygame.init()
        pygame.joystick.init()

        # ckeck if the gamepad is connected
        assert pygame.joystick.get_count() != 0, ("Gamepad not found. Please connect the gamepad and try again.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Gamepad has found: {self.joystick.get_name()}")
        logging.info(f"Number of gamepad: {pygame.joystick.get_count()}")

    def command_introduction(self):
        logging.info("\nMove <LS> to move the end effector along the x/y-axis.")
        logging.info("Press <LT/RT> to move the end effector along the z-axis.")
        logging.info("Move <RS> to rotate the end effector along the x/y-axis.")
        logging.info("Press <LT/RT + RS> to rotate the end effector along the z-axis.")
        logging.info("Press <RB> to open/close the gripper.")
        logging.info("Press <X> to switch the agent.")
        logging.info("Press <ESC> to exit.\n")

    def listen(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_id = event.axis
                axis_value = self.joystick.get_axis(axis_id)

            if event.type == pygame.JOYBUTTONDOWN:
                button_id = event.button
                if button_id == 2:
                    self._agent_id = 0 if self._agent_id else 1
                    logging.info(f"Switching agent to: {self._agent_id}")
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

        pos_increment = np.array([
            axis_value["axis_1"], 
            axis_value["axis_0"], 
            0.5 * (axis_value["axis_5"] - axis_value["axis_2"]) * (1 - button_value["button_10"])
        ]) * self.pos_scale

        rot_increment_x = axis_value["axis_3"]
        rot_increment_y = -axis_value["axis_4"]
        rot_increment_z = 0.5 * (axis_value["axis_5"] - axis_value["axis_2"]) * button_value["button_10"]
        rot_increment = np.array([rot_increment_x, rot_increment_y, rot_increment_z]) * self.rot_scale

        return (
            np.clip(pos_increment, -0.04, 0.04),
            T.euler_2_mat(rot_increment)
        )

    def __del__(self):
        pygame.joystick.quit()
        pygame.quit()
