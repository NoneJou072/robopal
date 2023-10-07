import robopal.commons.transform as trans
import numpy as np
import time


def test_transform():
    per_quat_2_mat_time = time.time()
    for i in range(1000):
        trans.quat_2_mat(np.array([1, 0, 0, 0]))
    print("per_quat_2_mat_time: ", time.time() - per_quat_2_mat_time)


if __name__ == '__main__':
    test_transform()
