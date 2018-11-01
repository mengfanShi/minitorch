"""
Optimizer:
SGD
Momentum
Adam
"""
import numpy as np


class SGD:
    def __init__(self, lr=1e-3):
        self.lr = lr

    def setlr(self, lr):
        self.lr = lr

    def optimize(self, grad, name):
        return self.lr * grad


class Momentum:
    def __init__(self, momentum=0.9, lr=1e-3):
        self.momentum = momentum
        self.lr = lr
        self.middle = dict()

    def setlr(self, lr):
        self.lr = lr

    def setmomentum(self, momentum):
        self.momentum = momentum

    def optimize(self, grad, name):
        cal = self.middle.get(name, np.zeros_like(grad))   # if middle name exists in the dict, load it; else create it
        cal = self.momentum * cal - self.lr * grad
        self.middle[name] = cal
        return -cal


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, lr=1e-3, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps
        self.middle = dict()

    def setbeta1(self, beta1):
        self.beta1 = beta1

    def setbeta2(self, beta2):
        self.beta2 = beta2

    def setlr(self, lr):
        self.lr = lr

    def optimize(self, grad, name):
        # initialize or load
        m = self.middle.get(name+':m', np.zeros_like(grad))
        v = self.middle.get(name+':v', np.zeros_like(grad))
        t = self.middle.get(name+':t', 0)

        # update
        t = t + 1
        m = m * self.beta1 + grad * (1 - self.beta1)
        v = v * self.beta2 + (grad ** 2) * (1 - self.beta2)
        self.middle[name+':m'] = m
        self.middle[name+':v'] = v
        self.middle[name+':t'] = t

        m_ = m / (1 - self.beta1 ** t)
        v_ = v / (1 - self.beta2 ** t)
        return self.lr * m_ / (np.sqrt(v_) + self.eps)
