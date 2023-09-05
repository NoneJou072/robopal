import numpy as np
from math import *


def isosqrt(var):
    if var < 1e-8:
        return 0
    else:
        return var


def isolate(var):
    if var < 1e-8:
        return 1e-8
    else:
        return var


def Update_Jaco(th):
    th1 = th[0]
    th2 = th[1]
    th3 = th[2]
    th4 = th[3]
    th5 = th[4]
    th6 = th[5]
    th7 = th[6]

    Jaco = np.array([
        [
            -0.1059 * (((-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                th5)) * sin(th6) - 0.076728 * (((-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                           1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                                                       th3)) * sin(th5)) * cos(th6) - 0.076728 * (
                        1.0 * (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6) + 0.1059 * (
                        1.0 * (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * cos(th6) - 0.005 * (
                        (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * sin(th5) + 0.005 * (
                        1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5) + 0.45669 * (
                        -sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 0.052254 * (
                        -sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) + 0.052254 * sin(
                th1) * sin(th2) * sin(th4) + 0.45669 * sin(th1) * sin(th2) * cos(th4) + 0.45889 * sin(th1) * sin(
                th2) - 0.064454 * sin(th1) * cos(th2) * cos(th3) + 0.064454 * sin(th3) * cos(th1) + 0.0005 * cos(th1)
            ,
            -0.1059 * ((-sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * cos(
                th5) + 1.0 * sin(th2) * sin(th3) * sin(th5) * cos(th1)) * sin(th6) - 0.076728 * (
                        (-sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5) * cos(th1)) * cos(th6) - 0.076728 * (
                        -1.0 * sin(th2) * sin(th4) * cos(th1) * cos(th3) - 1.0 * cos(th1) * cos(th2) * cos(th4)) * sin(
                th6) + 0.1059 * (
                        -1.0 * sin(th2) * sin(th4) * cos(th1) * cos(th3) - 1.0 * cos(th1) * cos(th2) * cos(th4)) * cos(
                th6) - 0.005 * (
                        -sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * sin(
                th5) + 0.005 * sin(th2) * sin(th3) * cos(th1) * cos(th5) - 0.45669 * sin(th2) * sin(th4) * cos(
                th1) * cos(th3) + 0.052254 * sin(th2) * cos(th1) * cos(th3) * cos(th4) - 0.064454 * sin(th2) * cos(
                th1) * cos(th3) - 0.052254 * sin(th4) * cos(th1) * cos(th2) - 0.45669 * cos(th1) * cos(th2) * cos(
                th4) - 0.45889 * cos(th1) * cos(th2)
            ,
            -0.1059 * ((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th5) + (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) * cos(th5)) * sin(
                th6) - 0.076728 * ((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th5) + (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) * cos(th5)) * cos(
                th6) + 0.005 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(
                th5) - 0.076728 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) * sin(
                th6) + 0.1059 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) * cos(
                th6) + 0.45669 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) - 0.005 * (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th5) * cos(th4) - 0.052254 * (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) + 0.064454 * sin(
                th1) * cos(th3) - 0.064454 * sin(th3) * cos(th1) * cos(th2)
            ,
            -0.005 * (-(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                th1) * cos(th4)) * sin(th5) - 0.1059 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * sin(th6) * cos(th5) - 0.076728 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * cos(th5) * cos(th6) - 0.076728 * (
                        1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * sin(th6) + 0.1059 * (
                        1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th6) + 0.052254 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 0.45669 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 0.45669 * sin(
                th2) * sin(th4) * cos(th1) - 0.052254 * sin(th2) * cos(th1) * cos(th4)
            ,
            -0.1059 * (-((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * sin(th5) + (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(
                th5)) * sin(th6) - 0.076728 * (-(
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * cos(th6) - 0.005 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * cos(th5) - 0.005 * (
                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(th5)
            ,
            0.076728 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * cos(th5) + (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * sin(th6) - 0.1059 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                         1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                     th2)) * sin(th5)) * cos(th6) - 0.1059 * (
                        1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6) - 0.076728 * (
                        1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * cos(th6)
            ,
            0
        ],
        [
            -0.1059 * (((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                   -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(th5)) * sin(
                th6) - 0.076728 * (((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(
                th4) - 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                               -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) - 0.076728 * (
                        1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) + 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6) + 0.1059 * (
                        1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) + 1.0 * sin(th2) * cos(th1) * cos(th4)) * cos(th6) - 0.005 * (
                        (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * sin(th5) + 0.45669 * (
                        -1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 0.052254 * (
                        -1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 0.005 * (
                        -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(th5) - 0.064454 * sin(
                th1) * sin(th3) - 0.0005 * sin(th1) + 0.052254 * sin(th2) * sin(th4) * cos(th1) + 0.45669 * sin(
                th2) * cos(th1) * cos(th4) + 0.45889 * sin(th2) * cos(th1) - 0.064454 * cos(th1) * cos(th2) * cos(th3)
            ,
            -0.1059 * ((1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * cos(
                th5) - 1.0 * sin(th1) * sin(th2) * sin(th3) * sin(th5)) * sin(th6) - 0.076728 * (
                        (1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * cos(
                    th5) - 1.0 * sin(th1) * sin(th2) * sin(th3) * sin(th5)) * cos(th6) - 0.076728 * (
                        1.0 * sin(th1) * sin(th2) * sin(th4) * cos(th3) + 1.0 * sin(th1) * cos(th2) * cos(th4)) * sin(
                th6) + 0.1059 * (
                        1.0 * sin(th1) * sin(th2) * sin(th4) * cos(th3) + 1.0 * sin(th1) * cos(th2) * cos(th4)) * cos(
                th6) - 0.005 * (
                        1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * sin(
                th5) - 0.005 * sin(th1) * sin(th2) * sin(th3) * cos(th5) + 0.45669 * sin(th1) * sin(th2) * sin(
                th4) * cos(th3) - 0.052254 * sin(th1) * sin(th2) * cos(th3) * cos(th4) + 0.064454 * sin(th1) * sin(
                th2) * cos(th3) + 0.052254 * sin(th1) * sin(th4) * cos(th2) + 0.45669 * sin(th1) * cos(th2) * cos(
                th4) + 0.45889 * sin(th1) * cos(th2)
            ,
            -0.1059 * ((1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th4) * cos(th5) + (
                        1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * sin(th5)) * sin(
                th6) - 0.076728 * (
                        (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th4) * cos(th5) + (
                            1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * sin(th5)) * cos(
                th6) - 0.076728 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th4) * sin(
                th6) + 0.1059 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th4) * cos(
                th6) + 0.45669 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                th4) - 0.005 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5) * cos(
                th4) - 0.052254 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(
                th4) + 0.005 * (1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * cos(
                th5) + 0.064454 * sin(th1) * sin(th3) * cos(th2) + 0.064454 * cos(th1) * cos(th3)
            ,
            -0.005 * (-(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                th1) * sin(th2) * cos(th4)) * sin(th5) - 0.1059 * (
                        -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6) * cos(th5) - 0.076728 * (
                        -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * cos(th5) * cos(th6) - 0.076728 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(
                    th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * sin(th6) + 0.1059 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(
                    th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th6) + 0.052254 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 0.45669 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 0.45669 * sin(
                th1) * sin(th2) * sin(th4) + 0.052254 * sin(th1) * sin(th2) * cos(th4)
            ,
            -0.1059 * (-((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * sin(th5) + (
                                   1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * sin(
                th6) - 0.076728 * (-(
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) + (
                                               1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(
                th5)) * cos(th6) - 0.005 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * cos(th5) - 0.005 * (
                        1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)
            ,
            0.076728 * (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                    1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * sin(
                th6) - 0.1059 * (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                             1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                th5)) * cos(th6) - 0.1059 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                    th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6) - 0.076728 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                    th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6)
            ,
            0
        ],
        [
            0
            ,
            0.1059 * (1.0 * (-1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * cos(th5) + 1.0 * sin(
                th3) * sin(th5) * cos(th2)) * sin(th6) + 0.076728 * (
                        1.0 * (-1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * cos(
                    th5) + 1.0 * sin(th3) * sin(th5) * cos(th2)) * cos(th6) + 0.005 * (
                        -1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * sin(th5) - 0.076728 * (
                        -1.0 * sin(th2) * cos(th4) + 1.0 * sin(th4) * cos(th2) * cos(th3)) * sin(th6) + 0.1059 * (
                        -1.0 * sin(th2) * cos(th4) + 1.0 * sin(th4) * cos(th2) * cos(th3)) * cos(th6) - 0.052254 * sin(
                th2) * sin(th4) - 0.45669 * sin(th2) * cos(th4) - 0.45889 * sin(th2) - 0.005 * sin(th3) * cos(
                th2) * cos(th5) + 0.45669 * sin(th4) * cos(th2) * cos(th3) - 0.052254 * cos(th2) * cos(th3) * cos(
                th4) + 0.064454 * cos(th2) * cos(th3)
            ,
            0.1059 * (1.0 * sin(th2) * sin(th3) * cos(th4) * cos(th5) + 1.0 * sin(th2) * sin(th5) * cos(th3)) * sin(
                th6) + 0.076728 * (
                        1.0 * sin(th2) * sin(th3) * cos(th4) * cos(th5) + 1.0 * sin(th2) * sin(th5) * cos(th3)) * cos(
                th6) + 0.076728 * sin(th2) * sin(th3) * sin(th4) * sin(th6) - 0.1059 * sin(th2) * sin(th3) * sin(
                th4) * cos(th6) - 0.45669 * sin(th2) * sin(th3) * sin(th4) + 0.005 * sin(th2) * sin(th3) * sin(
                th5) * cos(th4) + 0.052254 * sin(th2) * sin(th3) * cos(th4) - 0.064454 * sin(th2) * sin(
                th3) - 0.005 * sin(th2) * cos(th3) * cos(th5)
            ,
            0.005 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th5) + 0.1059 * (
                        1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) * cos(
                th5) + 0.076728 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th5) * cos(
                th6) - 0.076728 * (1.0 * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th4) * cos(th2)) * sin(
                th6) + 0.1059 * (1.0 * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th4) * cos(th2)) * cos(
                th6) + 0.052254 * sin(th2) * sin(th4) * cos(th3) + 0.45669 * sin(th2) * cos(th3) * cos(
                th4) - 0.45669 * sin(th4) * cos(th2) + 0.052254 * cos(th2) * cos(th4)
            ,
            0.1059 * (-1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(th5) + 1.0 * sin(
                th2) * sin(th3) * cos(th5)) * sin(th6) + 0.076728 * (
                        -1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                    th5) + 1.0 * sin(th2) * sin(th3) * cos(th5)) * cos(th6) + 0.005 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 0.005 * sin(
                th2) * sin(th3) * sin(th5)
            ,
            -0.076728 * (1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * sin(th6) + 0.1059 * (
                        1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) - 0.1059 * (
                        1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) - 0.076728 * (
                        1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6)
            ,
            0
        ],
        [
            0.5 * ((1.0 * (((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                       -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (
                                1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(
                            th4) + 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) / 2 - 0.5 * (1.0 * (((
                                                                                                                                 -sin(
                                                                                                                                     th1) * cos(
                                                                                                                             th2) * cos(
                                                                                                                             th3) + 1.0 * sin(
                                                                                                                             th3) * cos(
                                                                                                                             th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(
                th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                        -sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(th1) * sin(
                th2) * cos(th4)) * sin(th6)) * cos(th7) - 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * sin(th5) - 1.0 * (-1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(
                th1) * cos(th2)) * cos(th5)) * cos(th7) + (-1.0 * (
                        (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * sin(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                th3)) * cos(th5)) * sin(th7) / 2) / isolate(sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                    1.0 * sin(
                                                                                                th1) * cos(
                                                                                                th3) - 1.0 * sin(
                                                                                                th3) * cos(th1) * cos(
                                                                                                th2)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                         1.0 * sin(th1) * sin(
                                                                                     th3) * cos(th2) + 1.0 * cos(
                                                                                     th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * ((1.0 * (
                        (1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * cos(
                    th5) - 1.0 * sin(th1) * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 1.0 * (
                                1.0 * sin(th1) * sin(th2) * sin(th4) * cos(th3) + 1.0 * sin(th1) * cos(th2) * cos(
                            th4)) * sin(th6)) * sin(th7) / 2 - 0.5 * (1.0 * (
                        (-sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5) * cos(th1)) * cos(th6) + 1.0 * (
                                                                                  -1.0 * sin(th2) * sin(th4) * cos(
                                                                              th1) * cos(th3) - 1.0 * cos(th1) * cos(
                                                                              th2) * cos(th4)) * sin(th6)) * cos(
                th7) - 0.5 * (1.0 * (-1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * cos(
                th5) + 1.0 * sin(th3) * sin(th5) * cos(th2)) * sin(th6) - 0.5 * (1.0 * (
                        1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * sin(
                th5) + 1.0 * sin(th1) * sin(th2) * sin(th3) * cos(th5)) * cos(th7) + (-1.0 * (
                        -sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * sin(
                th5) + 1.0 * sin(th2) * sin(th3) * cos(th1) * cos(th5)) * sin(th7) / 2 - 0.5 * (
                               -1.0 * sin(th2) * cos(th4) + 1.0 * sin(th4) * cos(th2) * cos(th3)) * cos(th6)) / isolate(
                sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                    th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (-0.5 * (1.0 * ((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th5) + (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) * cos(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) * sin(th6)) * cos(
                th7) + (1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th5) - 1.0 * (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th5) * cos(th4)) * sin(
                th7) / 2 + (1.0 * (
                        (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th4) * cos(th5) + (
                            1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th4) * sin(
                th6)) * sin(th7) / 2 - 0.5 * (
                               1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                           th5) * cos(th4) - 1.0 * (
                                           1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * cos(
                           th5)) * cos(th7) - 0.5 * (
                               1.0 * sin(th2) * sin(th3) * cos(th4) * cos(th5) + 1.0 * sin(th2) * sin(th5) * cos(
                           th3)) * sin(th6) + 0.5 * sin(th2) * sin(th3) * sin(th4) * cos(th6)) / isolate(sqrt(isosqrt(
                -1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(
                    th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                    th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (
                            1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                        th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                            1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (-0.5 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * sin(th5) * sin(th7) - 0.5 * (
                               -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                           th4) - 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th5) * cos(th7) - 0.5 * (1.0 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * sin(th6)) * cos(th7) + (1.0 * (
                        -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * sin(th6)) * sin(th7) / 2 - 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) * cos(
                th5) - 0.5 * (1.0 * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th4) * cos(th2)) * cos(th6)) / isolate(
                sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                    th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (-0.5 * (-(
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + (
                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(
                th5)) * cos(th6) * cos(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * cos(th5) - 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * sin(th5)) * sin(th7) / 2 + 0.5 * (-(
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7) * cos(th6) - 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * cos(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * sin(th5)) * cos(th7) - 0.5 * (
                               -1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                           th5) + 1.0 * sin(th2) * sin(th3) * cos(th5)) * sin(th6)) / isolate(sqrt(isosqrt(-1.0 * (
                        1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(
                th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(
                th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (-0.5 * (-1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                               1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * sin(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * cos(th6)) * cos(th7) + (-1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * sin(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6)) * sin(th7) / 2 - 0.5 * (
                               1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                           th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6)) / isolate(
                sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                    th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (0.5 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                             1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) + (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * cos(th7) / 2 + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * cos(th7) / 2 + 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7)) / isolate(sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                           1.0 * sin(
                                                                                                       th1) * cos(
                                                                                                       th3) - 1.0 * sin(
                                                                                                       th3) * cos(
                                                                                                       th1) * cos(
                                                                                                       th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
        ],
        [
            0.5 * (-(1.0 * (((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                        -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (
                                 1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(
                             th4) + 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) / 2 - 0.5 * (1.0 * (((
                                                                                                                                  -sin(
                                                                                                                                      th1) * cos(
                                                                                                                              th2) * cos(
                                                                                                                              th3) + 1.0 * sin(
                                                                                                                              th3) * cos(
                                                                                                                              th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(
                th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                        -sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(th1) * sin(
                th2) * cos(th4)) * sin(th6)) * cos(th7) + 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * sin(th5) - 1.0 * (-1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(
                th1) * cos(th2)) * cos(th5)) * cos(th7) + (-1.0 * (
                        (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * sin(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                th3)) * cos(th5)) * sin(th7) / 2) / isolate(sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                    1.0 * sin(
                                                                                                th1) * cos(
                                                                                                th3) - 1.0 * sin(
                                                                                                th3) * cos(th1) * cos(
                                                                                                th2)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                         1.0 * sin(th1) * sin(
                                                                                     th3) * cos(th2) + 1.0 * cos(
                                                                                     th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (-(1.0 * (
                        (1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * cos(
                    th5) - 1.0 * sin(th1) * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 1.0 * (
                                 1.0 * sin(th1) * sin(th2) * sin(th4) * cos(th3) + 1.0 * sin(th1) * cos(th2) * cos(
                             th4)) * sin(th6)) * sin(th7) / 2 - 0.5 * (1.0 * (
                        (-sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5) * cos(th1)) * cos(th6) + 1.0 * (
                                                                                   -1.0 * sin(th2) * sin(th4) * cos(
                                                                               th1) * cos(th3) - 1.0 * cos(th1) * cos(
                                                                               th2) * cos(th4)) * sin(th6)) * cos(
                th7) + 0.5 * (1.0 * (-1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * cos(
                th5) + 1.0 * sin(th3) * sin(th5) * cos(th2)) * sin(th6) + 0.5 * (1.0 * (
                        1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * sin(
                th5) + 1.0 * sin(th1) * sin(th2) * sin(th3) * cos(th5)) * cos(th7) + (-1.0 * (
                        -sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * sin(
                th5) + 1.0 * sin(th2) * sin(th3) * cos(th1) * cos(th5)) * sin(th7) / 2 + 0.5 * (
                               -1.0 * sin(th2) * cos(th4) + 1.0 * sin(th4) * cos(th2) * cos(th3)) * cos(th6)) / isolate(
                sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                    th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (-0.5 * (1.0 * ((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th5) + (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) * cos(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) * sin(th6)) * cos(
                th7) + (1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th5) - 1.0 * (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th5) * cos(th4)) * sin(
                th7) / 2 - (1.0 * (
                        (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th4) * cos(th5) + (
                            1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th4) * sin(
                th6)) * sin(th7) / 2 + 0.5 * (
                               1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                           th5) * cos(th4) - 1.0 * (
                                           1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * cos(
                           th5)) * cos(th7) + 0.5 * (
                               1.0 * sin(th2) * sin(th3) * cos(th4) * cos(th5) + 1.0 * sin(th2) * sin(th5) * cos(
                           th3)) * sin(th6) - 0.5 * sin(th2) * sin(th3) * sin(th4) * cos(th6)) / isolate(sqrt(isosqrt(
                -1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(
                    th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                    th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (
                            1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                        th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                            1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (-0.5 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * sin(th5) * sin(th7) + 0.5 * (
                               -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                           th4) - 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th5) * cos(th7) - 0.5 * (1.0 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * sin(th6)) * cos(th7) - (1.0 * (
                        -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * sin(th6)) * sin(th7) / 2 + 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) * cos(
                th5) + 0.5 * (1.0 * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th4) * cos(th2)) * cos(th6)) / isolate(
                sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                    th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (-0.5 * (-(
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + (
                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(
                th5)) * cos(th6) * cos(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * cos(th5) - 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * sin(th5)) * sin(th7) / 2 - 0.5 * (-(
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7) * cos(th6) + 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * cos(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * sin(th5)) * cos(th7) + 0.5 * (
                               -1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                           th5) + 1.0 * sin(th2) * sin(th3) * cos(th5)) * sin(th6)) / isolate(sqrt(isosqrt(-1.0 * (
                        1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(
                th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(
                th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (-0.5 * (-1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                               1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * sin(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * cos(th6)) * cos(th7) - (-1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * sin(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6)) * sin(th7) / 2 + 0.5 * (
                               1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                           th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) - 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6)) / isolate(
                sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                    th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (0.5 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                             1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) - (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * cos(th7) / 2 + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * cos(th7) / 2 - 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7)) / isolate(sqrt(isosqrt(-1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                           1.0 * sin(
                                                                                                       th1) * cos(
                                                                                                       th3) - 1.0 * sin(
                                                                                                       th3) * cos(
                                                                                                       th1) * cos(
                                                                                                       th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
        ],
        [
            0.5 * ((1.0 * (((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                       -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (
                                1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(
                            th4) + 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) / 2 + 0.5 * (1.0 * (((
                                                                                                                                 -sin(
                                                                                                                                     th1) * cos(
                                                                                                                             th2) * cos(
                                                                                                                             th3) + 1.0 * sin(
                                                                                                                             th3) * cos(
                                                                                                                             th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(
                th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                        -sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(th1) * sin(
                th2) * cos(th4)) * sin(th6)) * cos(th7) - 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * sin(th5) - 1.0 * (-1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(
                th1) * cos(th2)) * cos(th5)) * cos(th7) - (-1.0 * (
                        (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * sin(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                th3)) * cos(th5)) * sin(th7) / 2) / isolate(sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                   1.0 * sin(th1) * cos(
                                                                                               th3) - 1.0 * sin(
                                                                                               th3) * cos(th1) * cos(
                                                                                               th2)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                         1.0 * sin(th1) * sin(
                                                                                     th3) * cos(th2) + 1.0 * cos(
                                                                                     th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * ((1.0 * (
                        (1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * cos(
                    th5) - 1.0 * sin(th1) * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 1.0 * (
                                1.0 * sin(th1) * sin(th2) * sin(th4) * cos(th3) + 1.0 * sin(th1) * cos(th2) * cos(
                            th4)) * sin(th6)) * sin(th7) / 2 + 0.5 * (1.0 * (
                        (-sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5) * cos(th1)) * cos(th6) + 1.0 * (
                                                                                  -1.0 * sin(th2) * sin(th4) * cos(
                                                                              th1) * cos(th3) - 1.0 * cos(th1) * cos(
                                                                              th2) * cos(th4)) * sin(th6)) * cos(
                th7) + 0.5 * (1.0 * (-1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * cos(
                th5) + 1.0 * sin(th3) * sin(th5) * cos(th2)) * sin(th6) - 0.5 * (1.0 * (
                        1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * sin(
                th5) + 1.0 * sin(th1) * sin(th2) * sin(th3) * cos(th5)) * cos(th7) - (-1.0 * (
                        -sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * sin(
                th5) + 1.0 * sin(th2) * sin(th3) * cos(th1) * cos(th5)) * sin(th7) / 2 + 0.5 * (
                               -1.0 * sin(th2) * cos(th4) + 1.0 * sin(th4) * cos(th2) * cos(th3)) * cos(th6)) / isolate(
                sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                   th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(

             th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (0.5 * (1.0 * ((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th5) + (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) * cos(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) * sin(th6)) * cos(
                th7) - (1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th5) - 1.0 * (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th5) * cos(th4)) * sin(
                th7) / 2 + (1.0 * (
                        (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th4) * cos(th5) + (
                            1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th4) * sin(
                th6)) * sin(th7) / 2 - 0.5 * (
                               1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                           th5) * cos(th4) - 1.0 * (
                                           1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * cos(
                           th5)) * cos(th7) + 0.5 * (
                               1.0 * sin(th2) * sin(th3) * cos(th4) * cos(th5) + 1.0 * sin(th2) * sin(th5) * cos(
                           th3)) * sin(th6) - 0.5 * sin(th2) * sin(th3) * sin(th4) * cos(th6)) / isolate(sqrt(isosqrt(
                1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                          1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(
                    th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                    th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (
                            1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                        th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                            1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (0.5 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * sin(th5) * sin(th7) - 0.5 * (
                               -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                           th4) - 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th5) * cos(th7) + 0.5 * (1.0 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * sin(th6)) * cos(th7) + (1.0 * (
                        -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * sin(th6)) * sin(th7) / 2 + 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) * cos(
                th5) + 0.5 * (1.0 * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th4) * cos(th2)) * cos(th6)) / isolate(
                sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                   th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (0.5 * (-(
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + (
                                      1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(
                th5)) * cos(th6) * cos(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * cos(th5) - 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * sin(th5)) * sin(th7) / 2 + 0.5 * (-(
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7) * cos(th6) - 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * cos(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * sin(th5)) * cos(th7) + 0.5 * (
                               -1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                           th5) + 1.0 * sin(th2) * sin(th3) * cos(th5)) * sin(th6)) / isolate(sqrt(isosqrt(1.0 * (
                        1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(
                th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(
                th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (0.5 * (-1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                              1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * sin(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * cos(th6)) * cos(th7) + (-1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * sin(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6)) * sin(th7) / 2 + 0.5 * (
                               1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                           th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) - 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6)) / isolate(
                sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                   th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (-0.5 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                              1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) + (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * cos(th7) / 2 - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * cos(th7) / 2 + 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7)) / isolate(sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                          1.0 * sin(
                                                                                                      th1) * cos(
                                                                                                      th3) - 1.0 * sin(
                                                                                                      th3) * cos(
                                                                                                      th1) * cos(
                                                                                                      th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) + 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
        ],
        [
            0.5 * (-(1.0 * (((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                        -1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (
                                 1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(
                             th4) + 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) / 2 + 0.5 * (1.0 * (((
                                                                                                                                  -sin(
                                                                                                                                      th1) * cos(
                                                                                                                              th2) * cos(
                                                                                                                              th3) + 1.0 * sin(
                                                                                                                              th3) * cos(
                                                                                                                              th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(
                th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                        -sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(th1) * sin(
                th2) * cos(th4)) * sin(th6)) * cos(th7) + 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th4) - 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * sin(th5) - 1.0 * (-1.0 * sin(th1) * cos(th3) + 1.0 * sin(th3) * cos(
                th1) * cos(th2)) * cos(th5)) * cos(th7) - (-1.0 * (
                        (-sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * sin(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                th3)) * cos(th5)) * sin(th7) / 2) / isolate(sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                   1.0 * sin(th1) * cos(
                                                                                               th3) - 1.0 * sin(
                                                                                               th3) * cos(th1) * cos(
                                                                                               th2)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                         1.0 * sin(th1) * sin(
                                                                                     th3) * cos(th2) + 1.0 * cos(
                                                                                     th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (-(1.0 * (
                        (1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * cos(
                    th5) - 1.0 * sin(th1) * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 1.0 * (
                                 1.0 * sin(th1) * sin(th2) * sin(th4) * cos(th3) + 1.0 * sin(th1) * cos(th2) * cos(
                             th4)) * sin(th6)) * sin(th7) / 2 + 0.5 * (1.0 * (
                        (-sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5) * cos(th1)) * cos(th6) + 1.0 * (
                                                                                   -1.0 * sin(th2) * sin(th4) * cos(
                                                                               th1) * cos(th3) - 1.0 * cos(th1) * cos(
                                                                               th2) * cos(th4)) * sin(th6)) * cos(
                th7) - 0.5 * (1.0 * (-1.0 * sin(th2) * sin(th4) - 1.0 * cos(th2) * cos(th3) * cos(th4)) * cos(
                th5) + 1.0 * sin(th3) * sin(th5) * cos(th2)) * sin(th6) + 0.5 * (1.0 * (
                        1.0 * sin(th1) * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th1) * sin(th4) * cos(th2)) * sin(
                th5) + 1.0 * sin(th1) * sin(th2) * sin(th3) * cos(th5)) * cos(th7) - (-1.0 * (
                        -sin(th2) * cos(th1) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th1) * cos(th2)) * sin(
                th5) + 1.0 * sin(th2) * sin(th3) * cos(th1) * cos(th5)) * sin(th7) / 2 - 0.5 * (
                               -1.0 * sin(th2) * cos(th4) + 1.0 * sin(th4) * cos(th2) * cos(th3)) * cos(th6)) / isolate(
                sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                   th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (0.5 * (1.0 * ((-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * sin(th5) + (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * cos(th4) * cos(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th4) * sin(th6)) * cos(
                th7) - (1.0 * (-1.0 * sin(th1) * sin(th3) - 1.0 * cos(th1) * cos(th2) * cos(th3)) * cos(th5) - 1.0 * (
                        1.0 * sin(th1) * cos(th3) - sin(th3) * cos(th1) * cos(th2)) * sin(th5) * cos(th4)) * sin(
                th7) / 2 - (1.0 * (
                        (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th4) * cos(th5) + (
                            1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th4) * sin(
                th6)) * sin(th7) / 2 + 0.5 * (
                               1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                           th5) * cos(th4) - 1.0 * (
                                           1.0 * sin(th1) * cos(th2) * cos(th3) - 1.0 * sin(th3) * cos(th1)) * cos(
                           th5)) * cos(th7) - 0.5 * (
                               1.0 * sin(th2) * sin(th3) * cos(th4) * cos(th5) + 1.0 * sin(th2) * sin(th5) * cos(
                           th3)) * sin(th6) + 0.5 * sin(th2) * sin(th3) * sin(th4) * cos(th6)) / isolate(sqrt(isosqrt(
                1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                          1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(
                    th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                    th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(
                    th3)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (
                            1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                        th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                            1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (0.5 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * sin(th5) * sin(th7) + 0.5 * (
                               -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                           th4) - 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th5) * cos(th7) + 0.5 * (1.0 * (
                        -(1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) + 1.0 * sin(th2) * cos(
                    th1) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * sin(th6)) * cos(th7) - (1.0 * (
                        -(-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * cos(th5) * cos(th6) + 1.0 * (1.0 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * sin(th6)) * sin(th7) / 2 - 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) * cos(
                th5) - 0.5 * (1.0 * sin(th2) * cos(th3) * cos(th4) - 1.0 * sin(th4) * cos(th2)) * cos(th6)) / isolate(
                sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                   th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (0.5 * (-(
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + (
                                      1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(
                th5)) * cos(th6) * cos(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * cos(th5) - 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * sin(th5)) * sin(th7) / 2 - 0.5 * (-(
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7) * cos(th6) + 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * cos(th5) + 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * sin(th5)) * cos(th7) - 0.5 * (
                               -1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                           th5) + 1.0 * sin(th2) * sin(th3) * cos(th5)) * sin(th6)) / isolate(sqrt(isosqrt(1.0 * (
                        1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                    th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                    th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(
                th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(
                th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
            ,
            0.5 * (0.5 * (-1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                              1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * sin(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * cos(th6)) * cos(th7) - (-1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * sin(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6)) * sin(th7) / 2 - 0.5 * (
                               1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                           th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 0.5 * (
                               1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6)) / isolate(
                sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                    th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                   th2)) * sin(th5)) * cos(th6) + 1.0 * (1.0 * (
                            1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(
                    th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                                                                             1.0 * sin(th1) * sin(
                                                                                         th3) * cos(th2) + 1.0 * cos(
                                                                                         th1) * cos(th3)) * sin(
                    th5)) * cos(th6) + 1.0 * (1.0 * (
                            -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) + 1.0 * sin(
                    th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                            (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                        th2) * sin(th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(
                    th3) * cos(th1) * cos(th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                            (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                        th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(
                    th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                            -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                    th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(
                    th6) + 1)))
            ,
            0.5 * (-0.5 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                              1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) - (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * cos(th7) / 2 - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * cos(th7) / 2 - 0.5 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7)) / isolate(sqrt(isosqrt(1.0 * (1.0 * (((1.0 * sin(th1) * sin(
                th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                                                                          1.0 * sin(
                                                                                                      th1) * cos(
                                                                                                      th3) - 1.0 * sin(
                                                                                                      th3) * cos(
                                                                                                      th1) * cos(
                                                                                                      th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) - (1.0 * (((-1.0 * sin(th1) * cos(
                th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(
                th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7) + 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7) - 1.0 * (1.0 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(
                th2) * cos(th4)) * cos(th6) + 1)))
        ]

    ], dtype='float64'
    )
    return Jaco


def Update_FK(th):
    th1 = th[0]
    th2 = th[1]
    th3 = th[2]
    th4 = th[3]
    th5 = th[4]
    th6 = th[5]
    th7 = th[6]
    T_FK = np.array([
        [
            -1.0 * (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(
                th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                       1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(
                th4) - 1.0 * sin(th2) * cos(th1) * cos(th4)) * sin(th6)) * cos(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * sin(th7)
            ,
            (1.0 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * cos(th5) + (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * cos(th6) + 1.0 * (
                         1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                     th2) * cos(th1) * cos(th4)) * sin(th6)) * sin(th7) + (-1.0 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 1.0 * (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                th2)) * cos(th5)) * cos(th7)
            ,
            (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(th4) * cos(
                th1)) * cos(th5) + (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(th5)) * sin(
                th6) - 1.0 * (1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                th2) * cos(th1) * cos(th4)) * cos(th6)
            ,
            -0.1059 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                th4) * cos(th1)) * cos(th5) + (1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * sin(
                th5)) * sin(th6) - 0.076728 * (((1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(
                th4) + 1.0 * sin(th2) * sin(th4) * cos(th1)) * cos(th5) + (
                                                           1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(
                                                       th2)) * sin(th5)) * cos(th6) - 0.076728 * (
                        1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * sin(th6) + 0.1059 * (
                        1.0 * (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 1.0 * sin(
                    th2) * cos(th1) * cos(th4)) * cos(th6) - 0.005 * (
                        (1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 1.0 * sin(th2) * sin(
                    th4) * cos(th1)) * sin(th5) + 0.45669 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * sin(th4) - 0.052254 * (
                        1.0 * sin(th1) * sin(th3) + cos(th1) * cos(th2) * cos(th3)) * cos(th4) + 0.005 * (
                        1.0 * sin(th1) * cos(th3) - 1.0 * sin(th3) * cos(th1) * cos(th2)) * cos(th5) + 0.064454 * sin(
                th1) * sin(th3) + 0.0005 * sin(th1) - 0.052254 * sin(th2) * sin(th4) * cos(th1) - 0.45669 * sin(
                th2) * cos(th1) * cos(th4) - 0.45889 * sin(th2) * cos(th1) + 0.064454 * cos(th1) * cos(th2) * cos(
                th3) + 8.7058e-5
        ],
        [
            -1.0 * (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                       1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                th5)) * cos(th6) + 1.0 * (
                                1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                            th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * cos(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * sin(th7)
            ,
            (1.0 * (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * cos(
                th6) + 1.0 * (1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6)) * sin(th7) - 1.0 * (1.0 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) - 1.0 * (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(
                th1) * cos(th3)) * cos(th5)) * cos(th7)
            ,
            (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(th1) * sin(
                th2) * sin(th4)) * cos(th5) + (1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                th5)) * sin(th6) - 1.0 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                    th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6)
            ,
            -0.1059 * (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                   1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(th5)) * sin(
                th6) - 0.076728 * (((-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(
                th4) - 1.0 * sin(th1) * sin(th2) * sin(th4)) * cos(th5) + (
                                               1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * sin(
                th5)) * cos(th6) - 0.076728 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                    th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * sin(th6) + 0.1059 * (
                        1.0 * (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(
                    th4) + 1.0 * sin(th1) * sin(th2) * cos(th4)) * cos(th6) - 0.005 * (
                        (-1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) - 1.0 * sin(
                    th1) * sin(th2) * sin(th4)) * sin(th5) + 0.005 * (
                        1.0 * sin(th1) * sin(th3) * cos(th2) + 1.0 * cos(th1) * cos(th3)) * cos(th5) + 0.45669 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * sin(th4) - 0.052254 * (
                        -1.0 * sin(th1) * cos(th2) * cos(th3) + 1.0 * sin(th3) * cos(th1)) * cos(th4) + 0.052254 * sin(
                th1) * sin(th2) * sin(th4) + 0.45669 * sin(th1) * sin(th2) * cos(th4) + 0.45889 * sin(th1) * sin(
                th2) - 0.064454 * sin(th1) * cos(th2) * cos(th3) + 0.064454 * sin(th3) * cos(th1) + 0.0005 * cos(
                th1) - 0.00063474
        ],
        [
            -1.0 * (-1.0 * (1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) + 1.0 * (
                                1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6)) * cos(
                th7) + (1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                th5) - 1.0 * sin(th2) * sin(th3) * cos(th5)) * sin(th7)
            ,
            (-1.0 * (1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * cos(th6) + 1.0 * (
                         1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6)) * sin(th7) + (
                        1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(
                    th5) - 1.0 * sin(th2) * sin(th3) * cos(th5)) * cos(th7)
            ,
            -1.0 * (1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) - 1.0 * (
                        1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6)
            ,
            0.1059 * (1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(th5) + 1.0 * sin(
                th2) * sin(th3) * sin(th5)) * sin(th6) + 0.076728 * (
                        1.0 * (-1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * cos(
                    th5) + 1.0 * sin(th2) * sin(th3) * sin(th5)) * cos(th6) - 0.076728 * (
                        1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * sin(th6) + 0.1059 * (
                        1.0 * sin(th2) * sin(th4) * cos(th3) + 1.0 * cos(th2) * cos(th4)) * cos(th6) + 0.005 * (
                        -1.0 * sin(th2) * cos(th3) * cos(th4) + 1.0 * sin(th4) * cos(th2)) * sin(th5) - 0.005 * sin(
                th2) * sin(th3) * cos(th5) + 0.45669 * sin(th2) * sin(th4) * cos(th3) - 0.052254 * sin(th2) * cos(
                th3) * cos(th4) + 0.064454 * sin(th2) * cos(th3) + 0.052254 * sin(th4) * cos(th2) + 0.45669 * cos(
                th2) * cos(th4) + 0.45889 * cos(th2) + 0.29942
        ]
    ], dtype='float64')
    return T_FK