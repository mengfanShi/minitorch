"""
several active functions:
ReLU
Sigmoid
LeakyReLU
Softmax
Tanh
"""
import numpy as np


class ReLU:
    def __init__(self):
        self.middle = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.middle = x > 0
        return self.middle * x

    def backward(self, grad_in):
        return grad_in * self.middle

    def __repr__(self):
        return "Active Function : ReLU"


class Sigmoid:
    def __init__(self):
        self.middle = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.middle = 1.0 / 1 + np.exp(-x)
        return self.middle

    def backward(self, grad_in):
        return grad_in * self.middle * (1 - self.middle)

    def __repr__(self):
        return "Active Function : Sigmoid"


class LeakyReLU:
    def __init__(self, slope = 0.1):
        self.middle = None
        self.slope = slope

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.middle = x > 0
        return self.middle * x + (1 - self.middle) * self.slope * x

    def backward(self, grad_in):
        return self.middle * grad_in + (1 - self.middle) * self.slope * grad_in

    def __repr__(self):
        return "Active Function : LeakyReLU"


class Softmax:
    def __init__(self):
        self.middle = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # default : rows represent the different samples
        #           cols represent the scores
        x = x - np.max(x, axis=1, keepdims=True)            # prevent the boom
        self.middle = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return self.middle

    def backward(self, grad_in):
        temp = np.sum(grad_in * self.middle, axis=1, keepdims=True)
        return self.middle * grad_in - self.middle * temp

    def __repr__(self):
        return "Active Function : Softmax"


class Tanh:
    def __init__(self):
        self.middle = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        self.middle = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.middle

    def backward(self, grad_in):
        return grad_in * (1 - self.middle ** 2)

    def __repr__(self):
        return "Active Function : Tanh"