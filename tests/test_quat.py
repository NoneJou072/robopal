import torch
import numpy as np
import numpy as np


def quaternion_invert(quaternion):
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return torch.tensor([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

initial_ball_rot = torch.tensor([1, 0, 0, 0])
l_hands_rot = torch.tensor([0.707, 0, 0, -0.707])

rot_error = quaternion_raw_multiply(quaternion_invert(l_hands_rot), initial_ball_rot)
rot_error_norm = 2 * torch.arccos(abs(rot_error[0]))
print(rot_error_norm)

def quaternion_to_axis_angle(q):
    w, x, y, z = q
    # 计算角度 theta
    theta = 2 * np.arccos(w)
    
    # 计算旋转轴
    sin_theta_over_2 = np.sqrt(1 - w**2)
    if sin_theta_over_2 == 0:
        # 处理 sin(theta/2) = 0 的情况，这种情况表示没有旋转
        return (np.array([1, 0, 0]), 0)
    else:
        axis = np.array([x, y, z]) / sin_theta_over_2
        return (axis, theta)
axis, angle = quaternion_to_axis_angle(rot_error)
print("旋转角（弧度）:", angle)

def quat_2_euler(quaternion):
    """
    Converts quaternion into euler angles(format in xyz).

    :param quaternion: 1*4 quaternion
    :return: 1*3 euler angles
    """
    w, x, y, z = quaternion
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))

    input_value = 2 * (w * y - z * x)
    input_value = np.clip(input_value, -1, 1)  # 将输入值限制在 [-1, 1] 范围内
    pitch = np.arcsin(input_value)

    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    euler = np.array([roll, pitch, yaw])
    return euler
print(quat_2_euler(rot_error))
