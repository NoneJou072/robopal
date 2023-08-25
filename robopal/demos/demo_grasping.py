import mujoco
import numpy as np
from robopal.envs.task_ctrl_env import PosCtrlEnv
import robopal.utils.KDL_utils.transform as T
from robopal.utils.ompl_base import TrajPlanning
import cv2


class MotionPlanEnv(PosCtrlEnv):
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
        self.planner = TrajPlanning(
            planning_time=1.0,
            interpolate_num=2
        )

        self.check_t = 0  # collision check times
        self.action = None

        can_list = ['coke_can', 'ceramic_can', 'glass_can']
        self.can_pos, self.can_quat, self.can_rotm = self.getObjPose(can_list)

    def isStateValid(self, state):
        # collision detect
        table_collision = False
        can_collision = False
        x_table, y_table, z_table = self.getTableCollision('table')
        x_can, y_can, z_can = self.getCanCollision('coke_can')
        # print("z_can = ",z_can)
        x, y, z = state[0], state[1], state[2]
        self.check_t += 1
        # print("state = ",state[0],state[1],state[2])
        # print(x_table,y_table,z_table)
        # if x < x_table[1] + 0.005 and x >= x_table[0] - 0.005:
        #     print("X in the table")
        # if y < y_table[1] + 0.005 and y >= y_table[0] - 0.005:
        #     print("Y in the table")
        # if z <= z_table[1] + 0.002 :
        #     print("Z in the table ")
        if x_table[1] + 0.005 > x >= x_table[0] - 0.005 and y_table[1] + 0.005 > y >= y_table[0] - 0.005 and z <= \
                z_table[1] + 0.002:
            table_collision = True
        else:
            pass
        if x_can[1] > x >= x_can[0] and y_can[1] > y >= y_can[0] and z_can[1] > z >= z_can[0]:
            can_collision = True
        else:
            pass
        if table_collision and can_collision:
            return False
        else:
            return True

    def getTableCollision(self, name):
        pos = self.mj_data.body(name).xpos
        quat = self.mj_data.body(name).xquat
        size = self.mj_model.geom(name).size
        x_range = [pos[0] - size[0], pos[0] + size[0]]
        y_range = [pos[1] - size[1], pos[1] + size[1]]
        z_range = [0 + 0.4 - 0.29, pos[2] + size[2] - 0.29 + 0.4]  # 0.29 for chassis height effect
        return x_range, y_range, z_range

    def getCanCollision(self, name):
        pos = self.mj_data.body(name).xpos
        quat = self.mj_data.body(name).xquat
        size = self.mj_model.geom(name).size
        x_range = [pos[0] - size[0], pos[0] + size[0]]
        y_range = [pos[1] - size[1], pos[1] + size[1]]
        z_range = [pos[2] - size[2] - 0.29,
                   pos[2] + size[2] - 0.29]  # 0.4 is the height of table presented by the world
        return x_range, y_range, z_range

    def getObjPose(self, name_list):
        pos = {}
        quat = {}
        mat = {}
        for name in name_list:
            pos[name] = self.mj_data.body(name).xpos.copy()
            pos[name][2] -= 0.21
            quat[name] = self.mj_data.body(name).xquat.copy()
            mat[name] = self.mj_data.body(name).xmat.copy()
        return pos, quat, mat

    def move(self, pos, quat):
        def checkArriveState(state):
            current_pos, current_mat = self.kdl_solver.getEeCurrentPose(self.robot.single_arm.arm_qpos)
            current_quat = T.mat2Quat(current_mat)
            error = np.sum(np.abs(state[:3] - current_pos)) + np.sum(np.abs(state[3:] - current_quat))
            if error <= 0.01:
                return True
            return False

        while True:
            cur_pos, _ = self.kdl_solver.getEeCurrentPose(self.robot.single_arm.arm_qpos)
            res, traj = self.planner.plan(cur_pos, pos, self.vel_cur[:3])
            self.action = np.concatenate((traj[1][:3], quat), axis=0)
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

    def camera_viewer(self):
        renderer = mujoco.Renderer(self.mj_model)
        while True:
            renderer.update_scene(self.mj_data, camera="0_gripper_cam")
            org = renderer.render()
            image = org[:, :, ::-1]
            cv2.imshow('RGB Image', image)
            cv2.waitKey(1)
            if self.viewer.is_running() is False:
                break


if __name__ == "__main__":
    from robopal.assets.robots.diana_med import DianaMed

    env = MotionPlanEnv(
        robot=DianaMed(),
        renderer="viewer",
        is_render=True,
        control_freq=200,
        is_interpolate=True,
        is_pd=True
    )
    env.reset()

    env.gripper_ctrl('open')
    env.move(env.can_pos['coke_can'], env.can_quat['coke_can'])
    env.gripper_ctrl('close')

    for t in range(int(1e6)):
        env.step(env.action)
        if env.is_render:
            env.render()
