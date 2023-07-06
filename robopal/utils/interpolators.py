from ruckig import InputParameter, OutputParameter, Result, Ruckig
import numpy as np
import PyKDL as KDL


class OTG:
    def __init__(self,
                 OTG_Dof=7,
                 control_cycle=0.001,
                 max_velocity=0.0,
                 max_acceleration=0.0,
                 max_jerk=0.0):
        self.OTG_Dof = OTG_Dof
        self.control_cycle = control_cycle
        self.otg = Ruckig(self.OTG_Dof, self.control_cycle)

        self.inp = InputParameter(self.OTG_Dof)
        self.out = OutputParameter(self.OTG_Dof)

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def setOTGParam(self, qpos, qvel):
        self.inp.current_position = qpos
        self.inp.current_velocity = qvel
        self.inp.current_acceleration = np.zeros(7)

        self.inp.target_position = np.zeros(7)
        self.inp.target_velocity = np.zeros(7)
        self.inp.target_acceleration = np.zeros(7)

        self.inp.max_velocity = self.max_velocity * np.ones(7)
        self.inp.max_acceleration = self.max_acceleration * np.ones(7)
        self.inp.max_jerk = self.max_jerk * np.ones(7)

    def updateInput(self, action):
        self.inp.target_position = action

    def updateState(self):
        self.otg.update(self.inp, self.out)
        q_target = self.out.new_position
        vel_target = self.out.new_velocity
        self.out.pass_to_input(self.inp)
        return q_target, vel_target


def cartesian_poly5_vel_plan(posa, posb, vel_a, vel_b, acc_a, acc_b, duration):
    """
    输入：
    posa:  起始位姿A(KDL::Frame)
    posb:  末端位姿A(KDL::Frame)
    vel_a: 起始笛卡尔速度(array)(6维)
    vel_b: 末端笛卡尔速度(array)(6维)
    acc_a: 起始笛卡尔加速度(array)(6维)
    acc_b: 末端笛卡尔加速度(array)(6维)
    duration: 期望运行时间(s)
    输出：
    Trajectory: 类,包含位置(六维XYZRPY,KDL::Frame)，速度(6维)，加速度(6维)
    """
    # Extract positions
    posa_xyz = np.array([posa.p[0], posa.p[1], posa.p[2]])
    posb_xyz = np.array([posb.p[0], posb.p[1], posb.p[2]])
    posa_rpy = np.array(posa.M.GetRPY())
    posb_rpy = np.array(posb.M.GetRPY())

    # Initialize time vector
    t = np.linspace(0, duration, 1000)

    # Initialize position, velocity, and acceleration vectors
    pos = np.zeros((len(t), 6))
    vel = np.zeros((len(t), 6))
    acc = np.zeros((len(t), 6))

    # Solve for coefficients of the fifth-degree polynomial
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [1, duration, duration ** 2, duration ** 3, duration ** 4, duration ** 5],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 2 * duration, 3 * duration ** 2, 4 * duration ** 3, 5 * duration ** 4],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 2, 6 * duration, 12 * duration ** 2, 20 * duration ** 3]
    ])

    b_x = np.array([posa_xyz[0], posb_xyz[0], vel_a[0], vel_b[0], acc_a[0], acc_b[0]])
    b_y = np.array([posa_xyz[1], posb_xyz[1], vel_a[1], vel_b[1], acc_a[1], acc_b[1]])
    b_z = np.array([posa_xyz[2], posb_xyz[2], vel_a[2], vel_b[2], acc_a[2], acc_b[2]])
    b_R = np.array([posa_rpy[0], posb_rpy[0], vel_a[3], vel_b[3], acc_a[3], acc_b[3]])
    b_P = np.array([posa_rpy[1], posb_rpy[1], vel_a[4], vel_b[4], acc_a[4], acc_b[4]])
    b_Y = np.array([posa_rpy[2], posb_rpy[2], vel_a[5], vel_b[5], acc_a[5], acc_b[5]])

    # Solve for coefficients
    c_x = np.linalg.solve(A, b_x)
    c_y = np.linalg.solve(A, b_y)
    c_z = np.linalg.solve(A, b_z)
    c_R = np.linalg.solve(A, b_R)
    c_P = np.linalg.solve(A, b_P)
    c_Y = np.linalg.solve(A, b_Y)

    # Calculate position, velocity, and acceleration
    for i in range(len(t)):
        pos[i, 0] = c_x[0] + c_x[1] * t[i] + c_x[2] * t[i] ** 2 + c_x[3] * t[i] ** 3 + c_x[4] * t[i] ** 4 + c_x[5] * t[
            i] ** 5
        pos[i, 1] = c_y[0] + c_y[1] * t[i] + c_y[2] * t[i] ** 2 + c_y[3] * t[i] ** 3 + c_y[4] * t[i] ** 4 + c_y[5] * t[
            i] ** 5
        pos[i, 2] = c_z[0] + c_z[1] * t[i] + c_z[2] * t[i] ** 2 + c_z[3] * t[i] ** 3 + c_z[4] * t[i] ** 4 + c_z[5] * t[
            i] ** 5
        pos[i, 3] = c_R[0] + c_R[1] * t[i] + c_R[2] * t[i] ** 2 + c_R[3] * t[i] ** 3 + c_R[4] * t[i] ** 4 + c_R[5] * t[
            i] ** 5
        pos[i, 4] = c_P[0] + c_P[1] * t[i] + c_P[2] * t[i] ** 2 + c_P[3] * t[i] ** 3 + c_P[4] * t[i] ** 4 + c_P[5] * t[
            i] ** 5
        pos[i, 5] = c_Y[0] + c_Y[1] * t[i] + c_Y[2] * t[i] ** 2 + c_Y[3] * t[i] ** 3 + c_Y[4] * t[i] ** 4 + c_Y[5] * t[
            i] ** 5

        vel[i, 0] = c_x[1] + 2 * c_x[2] * t[i] + 3 * c_x[3] * t[i] ** 2 + 4 * c_x[4] * t[i] ** 3 + 5 * c_x[5] * t[
            i] ** 4
        vel[i, 1] = c_y[1] + 2 * c_y[2] * t[i] + 3 * c_y[3] * t[i] ** 2 + 4 * c_y[4] * t[i] ** 3 + 5 * c_y[5] * t[
            i] ** 4
        vel[i, 2] = c_z[1] + 2 * c_z[2] * t[i] + 3 * c_z[3] * t[i] ** 2 + 4 * c_z[4] * t[i] ** 3 + 5 * c_z[5] * t[
            i] ** 4
        vel[i, 3] = c_R[1] + 2 * c_R[2] * t[i] + 3 * c_R[3] * t[i] ** 2 + 4 * c_R[4] * t[i] ** 3 + 5 * c_R[5] * t[
            i] ** 4
        vel[i, 4] = c_P[1] + 2 * c_P[2] * t[i] + 3 * c_P[3] * t[i] ** 2 + 4 * c_P[4] * t[i] ** 3 + 5 * c_P[5] * t[
            i] ** 4
        vel[i, 5] = c_Y[1] + 2 * c_Y[2] * t[i] + 3 * c_Y[3] * t[i] ** 2 + 4 * c_Y[4] * t[i] ** 3 + 5 * c_Y[5] * t[
            i] ** 4

        acc[i, 0] = 2 * c_x[2] + 6 * c_x[3] * t[i] + 12 * c_x[4] * t[i] ** 2 + 20 * c_x[5] * t[i] ** 3
        acc[i, 1] = 2 * c_y[2] + 6 * c_y[3] * t[i] + 12 * c_y[4] * t[i] ** 2 + 20 * c_y[5] * t[i] ** 3
        acc[i, 2] = 2 * c_z[2] + 6 * c_z[3] * t[i] + 12 * c_z[4] * t[i] ** 2 + 20 * c_z[5] * t[i] ** 3
        acc[i, 3] = 2 * c_R[2] + 6 * c_R[3] * t[i] + 12 * c_R[4] * t[i] ** 2 + 20 * c_R[5] * t[i] ** 3
        acc[i, 4] = 2 * c_P[2] + 6 * c_P[3] * t[i] + 12 * c_P[4] * t[i] ** 2 + 20 * c_P[5] * t[i] ** 3
        acc[i, 5] = 2 * c_Y[2] + 6 * c_Y[3] * t[i] + 12 * c_Y[4] * t[i] ** 2 + 20 * c_Y[5] * t[i] ** 3

    # Return trajectory
    return Trajectory(pos, vel, acc, duration)


class Trajectory:
    def __init__(self, pos, vel, acc, duration):
        self.pos_data = pos
        self.vel_data = vel
        self.acc_data = acc
        self.duration = duration

    def pos(self, t):
        return self.interpolate(self.pos_data, t)

    def vel(self, t):
        return self.interpolate(self.vel_data, t)

    def acc(self, t):
        return self.interpolate(self.acc_data, t)

    def interpolate(self, data, t):
        # 线性插值计算在给定时间点 t 处的值
        t_normalized = t / self.duration
        idx = int(np.floor(t_normalized * (len(data) - 1)))
        if idx < len(data) - 1:
            t_interp = t_normalized * (len(data) - 1) - idx
            return data[idx] + t_interp * (data[idx + 1] - data[idx])
        else:
            return data[-1]

    def to_frame(self, t):
        pos = self.pos(t)
        pos_xyz = pos[:3]
        pos_rpy = pos[3:]
        pos_frame = KDL.Frame(
            KDL.Rotation.RPY(pos_rpy[0], pos_rpy[1], pos_rpy[2]),
            KDL.Vector(pos_xyz[0], pos_xyz[1], pos_xyz[2])
        )
        return pos_frame


def run_trajectory():
    # 设置起点和终点位置
    posa = KDL.Frame(KDL.Rotation.RPY(0, 0, 0), KDL.Vector(0, 0, 0))
    posb = KDL.Frame(KDL.Rotation.RPY(np.pi / 2, np.pi / 4, np.pi / 8), KDL.Vector(1.5, 2, 2.5))
    vel_a = np.array([0.2, 0.3, 0.1, 0.4, 0.1, 0.1])
    vel_b = np.array([0, 0, 0, 0, 0, 0])
    acc_a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    acc_b = np.array([0, 0, 0, 0, 0, 0])
    # 设置运行时间
    duration = 2.0

    # 计算轨迹
    traj = cartesian_poly5_vel_plan(posa, posb, vel_a, vel_b, acc_a, acc_b, duration)
    print("目标位置：\n", posb)
    print("计算位置：\n", traj.to_frame(traj.duration))

# 运行测试程序
# run_trajectory()
