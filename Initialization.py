"""
parameters initialize:
Gaussio distribute
Constant initialize
Xavier initialize
He_normal initialize
"""
import numpy as np


class Initialize:
    def __init__(self, shape, withgrad=False, initializer=None):
        self.shape = shape
        self.data = None
        self.grad = None
        self.initializer = initializer
        self.initialize(withgrad)

    def initialize(self, withgrad=False):
        self.data = self.initializer(self.shape)
        if withgrad:
            self.grad = np.zeros(self.shape)

    def getdata(self):
        return self.data

    def getgrad(self):
        return self.grad

    def setdata(self, data):
        self.data = data

    def setgrad(self, grad):
        self.grad = grad


class GaussionInitialize:
    def __init__(self, mean=0, std=1e-2):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        return np.random.normal(self.mean, self.std, shape)


class ConstantInitialize:
    def __init__(self, constant=0.1):
        self.constant = constant

    def __call__(self, shape):
        return np.zeros(shape) + self.constant


# assume the active function attempt to be linear near the zero, also the active values is symmetric about 0 (like tanh)
class XavierInitialize:
    def __init__(self, mode='uniform'):
        self.mode = mode

    def __call__(self, shape):
        param = np.sum(shape)
        if self.mode == 'uniform':
            return np.random.uniform(-np.sqrt(6/param), np.sqrt(6/param), shape)
        else:
            return np.random.normal(0, np.sqrt(2/param), shape)


# used for relu
class HeInitialize:
    def __init__(self, mode='normal'):
        self.mode = mode

    def __call__(self, shape):
        if self.mode == 'normal':
            return np.random.normal(0, np.sqrt(2/shape[0]), shape)
        else:
            return np.random.uniform(-np.sqrt(6/shape[0]), np.sqrt(6/shape[0]), shape)







