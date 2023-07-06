import numpy as np
from math import *


class MyCustomException(Exception):
    pass


class Slerp:
    def __init__(self, Cart_S, Cart_E, N):
        self.Q4_S = self.cart_to_quat(Cart_S)
        self.Q4_E = self.cart_to_quat(Cart_E)
        self.sita = acos(np.dot(self.Q4_S, self.Q4_E))
        if self.sita < 0:
            self.Q4_E = -self.Q4_E
        self.N = N
        self.stept = 1 / self.N

    def cart_to_quat(self, T):
        Q4 = np.array([0.5 * sqrt(T[0, 0] + T[1, 1] + T[2, 2] + 1),
                       0.5 * (np.sign(T[2, 1] - T[1, 2])) * sqrt(T[0, 0] - T[1, 1] - T[2, 2] + 1),
                       0.5 * (np.sign(T[0, 2] - T[2, 0])) * sqrt(-T[0, 0] + T[1, 1] - T[2, 2] + 1),
                       0.5 * (np.sign(T[1, 0] - T[0, 1])) * sqrt(-T[0, 0] - T[1, 1] + T[2, 2] + 1)])
        return Q4

    def Equal(self, t):
        self.Q4_E = ((sin((1 - t) * self.sita)) / sin(self.sita)) * self.Q4_S + (
                    (sin((t) * self.sita)) / sin(self.sita)) * self.Q4_E
        return self.Q4_E

    def Slerp(self, t):
        try:
            if t < 1 or t > self.N:
                raise MyCustomException
            return self.Equal(t * self.stept)
        except MyCustomException:
            print("your index error!")
