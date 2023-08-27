import numpy as np
from numpy import sin, cos


def euler2Quat(rpy, degrees: bool = False):
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


def euler2Mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles
        rad args

    Returns:
        np.array: 3x3 rotation matrix

    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def quat2Mat(quaternion):
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


def vec2Mat(vec):
    '''
    The rotation vector modulus is the angle
    Not using the Rodriguez formula

    When calculating the rotation matrix,
    it is necessary to normalize the rotation vector into a unit vector,
    calculate the four components of the quaternion according to the formula,
    and then convert the quaternion into a rotation matrix.
    '''
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


def mat2Quat(mat):
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

    quaternion = np.array([qw, qx, qy, qz]) #w,x,y,z
    return quaternion


def makeTransform(pos=None, rot=None) -> np.ndarray:
    """ concatenate both 1*3 or 3*1 position array and 3*3 rotation matrix
        to a 4*4 transform matrix

    E.g.:
    >>> T = makeTransform(np.zeros(3), np.zeros(shape=(3, 3)))

    :param pos: 1*3 position
    :param rot: 3*3 rotation
    :return: 4*4 transform
    """
    pos = np.asarray(pos).reshape(3, 1)
    rot = np.asarray(rot)
    Trans = np.concatenate([rot, pos], axis=1)
    Trans = np.concatenate([Trans, np.array([[0, 0, 0, 1]])], axis=0)
    return Trans

# @jit(nopython=True)
def mat_transpose(mat):
    rows = len(mat)
    cols = len(mat[0])
    matrix_t = np.zeros(rows*cols).reshape(rows,cols)
    # matrix_t = [[0.0000 for _ in range(rows)] for _ in range(cols)]


    for i in range(rows):
        for j in range(cols):
            # print(mat[i][j])
            matrix_t[j][i] = mat[i][j]

    return matrix_t

