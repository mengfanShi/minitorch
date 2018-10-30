"""
Regularization:
BatchNormal
Dropout
"""
import numpy as np
import Initialization


class BatchNormal:
    def __init__(self, name, num_feature, mode='train', momentum=0.9, eps=1e-5):
        self.mode = mode
        self.momentum = momentum
        self.eps = eps
        self.name = name
        self.mean = np.zeros(shape=[num_feature], dtype='float32')
        self.var = np.zeros(shape=[num_feature], dtype='float32')
        self.middle = None

        self.param = dict()
        self.beta = Initialization.Initialize(shape=[num_feature], withgrad=True, initializer=Initialization.ConstantInitialize(0))
        self.gamma = Initialization.Initialize(shape=[num_feature], withgrad=True, initializer=Initialization.ConstantInitialize(1.0))
        self.param['beta'] = self.beta
        self.param['gamma'] = self.gamma

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if x.shape[0] == np.sum(x.shape):    # just one sample
            return x
        else:
            if self.mode == 'train':
                sample_mean = np.mean(x, axis=0)
                sample_var = np.var(x, axis=0)
                self.mean = self.mean * self.momentum + sample_mean * (1 - self.momentum)
                self.var = self.var * self.momentum + sample_var * (1 - self.momentum)
                std = np.sqrt(sample_var + self.eps)
                x_ = (x - sample_mean) / std
                out = self.gamma.getdata() * x_ + self.beta.getdata()
                self.middle = x_, std
            else:
                x_ = (x - self.mean) / np.sqrt(self.var + self.eps)
                out = self.gamma.getdata() * x_ + self.beta.getdata()
            return out

    def backward(self, grad_in):
        x_, std = self.middle
        if grad_in.shape[0] == np.sum(grad_in.shape):       # just one sample
            return grad_in
        else:
            num, dim = grad_in.shape

            dbeta = np.sum(grad_in, axis=0)
            dgamma = np.sum(grad_in * x_, axis=0)
            self.beta.setgrad(dbeta)
            self.gamma.setgrad(dgamma)

            middle = grad_in * self.gamma.getdata()
            dx = (1 - 1/num) * 1/std * (1 - x_**2)
            grad = middle * dx
            return grad

    def __repr__(self):
        return "Regularize : BatchNormal"


class Dropout:
    def __init__(self, name, mode='train', rate=0.5):
        self.mode = mode
        self.rate = rate
        self.name = name
        self.scale = 1 / rate
        self.middle = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        assert 1 - self.rate >= 0
        if self.mode == 'train':
            prob = np.random.rand(*x.shape)
            mask = prob > self.rate
            self.middle = mask
            return mask * x * self.scale
        else:
            return x

    def backward(self, grad_in):
        mask = self.middle
        grad = mask * grad_in * self.scale
        return grad

    def __repr__(self):
        return 'Regularize : Dropout({})'.format(self.rate)








