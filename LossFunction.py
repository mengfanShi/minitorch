"""
several loss functions:
CrossEntropy
L1 distance
L2 distance
"""
import numpy as np


class CrossEntropy:
    def __init__(self):
        self.middle = None

    def __call__(self, pred, label):
        return self.forward(pred, label)

    def forward(self, pred, label):
        # default : rows represent the different samples
        #           cols represent the scores
        assert pred.shape == label.shape
        result = (-1 / pred.shape[0]) * np.sum(label * np.log(pred))
        self.middle = pred, label
        return result

    def backward(self):
        pred, label = self.middle
        grad = (-1 / pred.shape[0]) * (label / pred)
        return grad

    def __repr__(self):
        return "Loss Function : CrossEntropy"


class L1:
    def __init__(self):
        self.middle = None

    def __call__(self, pred, label):
        return self.forward(pred, label)

    def forward(self, pred, label):
        assert pred.shape == label.shape
        result = np.mean(np.abs(pred - label))
        self.middle = pred, label
        return result

    def backward(self):
        pred, label = self.middle
        mask = (pred - label) >= 0
        grad = (1 / np.product(pred.shape)) * (2 * mask - 1)
        return grad

    def __repr__(self):
        return "Loss Function : L1"


class L2:
    def __init__(self):
        self.middle = None

    def __call__(self, pred, label):
        return self.forward(pred, label)

    def forward(self, pred, label):
        assert pred.shape == label.shape
        result = np.mean(np.square(pred - label))
        self.middle = pred, label
        return result

    def backward(self):
        pred, label = self.middle
        grad = (2 / np.product(pred.shape)) * (pred - label)
        return grad

    def __repr__(self):
        return "Loss Function : L2"