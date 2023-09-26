import numpy as np
from numpy import sin, cos


def euler_2_quat(rpy, degrees: bool = False):
    """
    degrees:bool
    True is the angle value, and False is the radian value
    """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([x, y, z, w])


def euler_2_mat(euler):
    """
    Converts euler angles(format in xyz) into rotation matrix form.

    :param euler: 1*3 euler angles
    :return: 3*3 rotation matrix
    """
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)
    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


def quat_2_mat(quaternion):
    """
    The rotation vector modulus is the angle
    Not using the Rodriguez formula

    :param quaternion:  1*4 quaternion
    :return:  3*3 rotation matrix
    """
    w, x, y, z = quaternion

    norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    r11 = 1 - 2 * y ** 2 - 2 * z ** 2
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y
    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x ** 2 - 2 * z ** 2
    r23 = 2 * y * z - 2 * w * x
    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x ** 2 - 2 * y ** 2

    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])

    return rotation_matrix


def vec2_mat(vec):
    """
    The rotation vector modulus is the angle
    Not using the Rodriguez formula

    When calculating the rotation matrix,
    it is necessary to normalize the rotation vector into a unit vector,
    calculate the four components of the quaternion according to the formula,
    and then convert the quaternion into a rotation matrix.
    """
    x = vec[0]
    y = vec[1]
    z = vec[2]
    theta = np.sqrt(x * x + y * y + z * z)
    axis = np.array([x, y, z]) / theta
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    bb, cc, dd = b ** 2, c ** 2, d ** 2
    bc, bd, cd = b * c, b * d, c * d
    rotation_matrix = np.array([[a * a + bb - cc - dd, 2 * (bc + a * d), 2 * (bd - a * c)],
                                [2 * (bc - a * d), a * a + cc - bb - dd, 2 * (cd + a * b)],
                                [2 * (bd + a * c), 2 * (cd - a * b), a * a + dd - bb - cc]])
    return rotation_matrix


def mat_2_vec(mat):
    """
    Converts rotation matrix into rotation vector form.

    :param mat: 3*3 rotation matrix
    :return: 1*3 rotation vector
    """
    theta = np.arccos((np.trace(mat) - 1) / 2)
    if theta == 0:
        return np.zeros(3)
    else:
        k = 1 / (2 * np.sin(theta))
        x = k * (mat[2, 1] - mat[1, 2]) * theta
        y = k * (mat[0, 2] - mat[2, 0]) * theta
        z = k * (mat[1, 0] - mat[0, 1]) * theta
        return np.array([x, y, z])


def mat_2_quat(mat):
    """
    The rotation vector modulus is the angle
    Not using the Rodriguez formula

    :param mat: 3*3 rotation matrix
    :return: 1*4 quaternion
    """
    trace = np.trace(mat)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4*qw
        qw = 0.25 * S
        qx = (mat[2, 1] - mat[1, 2]) / S
        qy = (mat[0, 2] - mat[2, 0]) / S
        qz = (mat[1, 0] - mat[0, 1]) / S
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        S = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2  # S = 4*qx
        qw = (mat[2, 1] - mat[1, 2]) / S
        qx = 0.25 * S
        qy = (mat[0, 1] + mat[1, 0]) / S
        qz = (mat[0, 2] + mat[2, 0]) / S
    elif mat[1, 1] > mat[2, 2]:
        S = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2  # S = 4*qy
        qw = (mat[0, 2] - mat[2, 0]) / S
        qx = (mat[0, 1] + mat[1, 0]) / S
        qy = 0.25 * S
        qz = (mat[1, 2] + mat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2  # S = 4*qz
        qw = (mat[1, 0] - mat[0, 1]) / S
        qx = (mat[0, 2] + mat[2, 0]) / S
        qy = (mat[1, 2] + mat[2, 1]) / S
        qz = 0.25 * S

    quaternion = np.array([qw, qx, qy, qz])  # w,x,y,z
    return quaternion


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


def vec_2_euler(vec):
    """
    Converts rotation vector into euler angles(format in xyz).

    :param vec: 1*3 rotation vector
    :return: 1*3 euler angles
    """
    mat = vec2_mat(vec)
    euler = mat_2_euler(mat)
    return euler


def mat_2_euler(mat):
    """
    Converts rotation matrix into euler angles(format in xyz).

    :param mat: 3*3 rotation matrix
    :return: 1*3 euler angles
    """
    euler = quat_2_euler(mat_2_quat(mat))
    return euler


def make_transform(pos=None, rot=None) -> np.ndarray:
    """ concatenate both 1*3 or 3*1 position array and 3*3 rotation matrix
        to a 4*4 transform matrix

    E.g.:
    >>> T = make_transform(np.zeros(3), np.zeros(shape=(3, 3)))

    :param pos: 1*3 position
    :param rot: 3*3 rotation
    :return: 4*4 transform
    """
    pos = np.asarray(pos).reshape(3, 1)
    rot = np.asarray(rot)
    return np.vstack((np.hstack((rot, pos)),
                      np.array([0, 0, 0, 1])))
