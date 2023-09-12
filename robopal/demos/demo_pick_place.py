import numpy as np
from robopal.envs.task_ik_ctrl_env import PosCtrlEnv
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

        can_list = ['green_block']
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

    def step(self, action):
        super().step(action)

    def gripper_ctrl(self, cmd: int):
        self.mj_data.actuator("0_gripper_l_finger_joint").ctrl = -20 if cmd == 0 else 20


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaGrasp

    env = PickAndPlaceEnv(
        robot=DianaGrasp(),
        renderer="viewer",
        is_render=True,
        control_freq=200,
        is_interpolate=False,
        is_pd=True
    )
    env.reset()

    for t in range(int(1e6)):
        action = np.array([0.4, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0])
        env.step(action)
        if env.is_render:
            env.render()
        if t % 1000 == 0:
            env.reset()
