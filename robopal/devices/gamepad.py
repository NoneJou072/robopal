import logging

try:
    import pygame
except ImportError:
    logging.warn("pygame is not installed. If you want to use the gamepad, Please install it by running 'pip install pygame'")

import numpy as np
from robopal.devices import BaseDevice


class Gamepad(BaseDevice):
    def __init__(self):
        self.joystick = None

    def start(self):
        self.command_introduction()

        # 初始化 Pygame
        pygame.init()

        # 初始化手柄
        pygame.joystick.init()

        # 检查是否有手柄连接
        if pygame.joystick.get_count() == 0:
            print("没有检测到手柄")
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"检测到手柄: {self.joystick.get_name()}")

    def command_introduction(self):
        logging.error("Not support now, please use other devices.")

    def listen(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # 读取轴的输入
                axis_id = event.axis
                axis_value = self.joystick.get_axis(axis_id)
                # print(f"轴 {axis_id} 值: {axis_value:.2f}")

            if event.type == pygame.JOYBUTTONDOWN:
                # 读取按键的输入
                for i in range(self.joystick.get_numbuttons()):
                    button = self.joystick.get_button(i)
                    # if button:
                    #     print(f"按键 {i} 按下")

            # if event.type == pygame.JOYBUTTONUP:
            #     # 按键释放
            #     print(f"按键 {event.button} 释放")

            if event.type == pygame.JOYHATMOTION:
                # 读取方向键输入
                for i in range(self.joystick.get_numhats()):
                    hat = self.joystick.get_hat(i)
                    # print(f"方向键 {i} 值: {hat}")

    def get_axis(self):
        """ 手动读取轴的输入 
        """
        self.listen()
        return {f"axis_{i}": self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())}
    
    def get_button(self):
        """ 手动读取按键的输入 
        """
        self.listen()
        return {f"button_{i}": self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())}
    
    def get_action_increment(self):
        self.listen()

        axis_value = self.get_axis()
        button_value = self.get_button()

        pos_increment_scale = 0.01
        pos_increment = np.array([
            axis_value["axis_0"], -axis_value["axis_1"], 0.5 * (axis_value["axis_5"] - axis_value["axis_2"]) * (1 - button_value["button_10"])
        ]) * pos_increment_scale

        rot_increment_scale = 0.1
        rot_increment_x = axis_value["axis_3"]
        rot_increment_y = -axis_value["axis_4"]
        rot_increment_z = 0.5 * (axis_value["axis_5"] - axis_value["axis_2"]) * button_value["button_10"]
        rot_increment = np.array([rot_increment_x, rot_increment_y, rot_increment_z]) * rot_increment_scale

        return pos_increment, rot_increment

    def __del__(self):
        pygame.joystick.quit()
        pygame.quit()
