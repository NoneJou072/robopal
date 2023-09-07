import numpy as np
from robopal.envs.task_ctrl_env import PosCtrlEnv
import robopal.commons.transform as T


class PickAndPlaceEnv(PosCtrlEnv):
    def __init__(self,
                 robot=None,
                 is_render=True,
                 renderer="viewer",
                 control_freq=20,
                 is_interpolate=True,
                 is_pd=True,
                 ):
        super().__init__(
            robot=robot,
            is_render=is_render,
            renderer=renderer,
            control_freq=control_freq,
            is_interpolate=is_interpolate,
            is_pd=is_pd
        )

        self.action = None

        can_list = ['red_block', 'blue_block', 'green_block']
        self.can_pos, self.can_quat, self.can_rotm = self.getObjPose(can_list)

    def getObjPose(self, name_list):
        pos = {}
        quat = {}
        mat = {}
        for name in name_list:
            pos[name] = self.mj_data.body(name).xpos.copy()
            pos[name][2] -= 0.33
            quat[name] = self.mj_data.body(name).xquat.copy()
            mat[name] = self.mj_data.body(name).xmat.copy()
        return pos, quat, mat

    def move(self, pos, quat):
        def checkArriveState(state):
            current_pos, current_mat = self.kdl_solver.fk(self.robot.single_arm.arm_qpos)
            current_quat = T.mat_2_quat(current_mat)
            error = np.sum(np.abs(state[:3] - current_pos)) + np.sum(np.abs(state[3:] - current_quat))
            if error <= 0.01:
                return True
            return False

        while True:
            self.action = np.concatenate((pos, quat), axis=0)
            self.step(self.action)
            if self.is_render:
                self.render()
            if checkArriveState(self.action):
                break

    def gripper_ctrl(self, cmd: str):
        if cmd == "open":
            self.mj_data.actuator("0_gripper_l_finger_joint").ctrl = 20
        elif cmd == "close":
            self.mj_data.actuator("0_gripper_l_finger_joint").ctrl = -20


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaGrasp

    env = PickAndPlaceEnv(
        robot=DianaGrasp(),
        renderer="viewer",
        is_render=True,
        control_freq=200,
        is_interpolate=True,
        is_pd=True
    )
    env.reset()

    env.gripper_ctrl('open')
    env.move(env.can_pos['blue_block'], env.can_quat['blue_block'])
    env.gripper_ctrl('close')
    env.move(env.can_pos['blue_block'] + np.array([0, 0, 0.1]), env.can_quat['blue_block'])

    for t in range(int(1e6)):
        env.step(env.action)
        if env.is_render:
            env.render()
