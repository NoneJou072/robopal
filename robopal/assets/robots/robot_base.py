import numpy as np
import os
import sys


class ModelBase(object):
    def __init__(self):
        self.arms = []
        self.chasis = None
        self.grippers = []
        pass

    def splicer(self):
        pass


    class arm(ArmBase):
        pass

    class mobile(MobileBase):
        pass

    class gripper(GripperBase):
        pass