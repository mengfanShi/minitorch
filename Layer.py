"""
Layer:
FullyConnect
Convolution
"""
import numpy as np
import Initialization


class FullyConnect:
    def __init__(self, name, feature_in, feature_out, mode='train', reg=1e-4, weight_init=None, bias_init=None):
        self.name = name
        self.input = feature_in
        self.output = feature_out
        self.weight = None
        self.bias = None
        self.middle = None
        self.reg = reg
        self.mode = mode

        self.param = dict()
        self.weight_init = weight_init
        self.bias_init = bias_init

    def __call__(self, x):
        return self.forward(x)

    def setmode(self, mode):
        self.mode = mode

    def setreg(self, reg):
        self.reg = reg

    def initialize(self):
        # judge the mode
        withgrad = False
        if self.mode == 'train':
            withgrad = True

        # init parameters
        if self.weight_init is None:
            self.weight_init = Initialization.GaussionInitialize(mean=0, std=1e-2)
        if self.bias_init is None:
            self.bias_init = Initialization.ConstantInitialize(constant=0.1)

        self.weight = Initialization.Initialize(shape=(self.input, self.output),
                                                withgrad=withgrad,
                                                initializer=self.weight_init)

        self.bias = Initialization.Initialize(shape=[self.output],
                                              withgrad=withgrad,
                                              initializer=self.bias_init)
        self.param['weight'] = self.weight
        self.param['bias'] = self.bias

    def forward(self, x):
        # x:[N, D1]   W:[D1, D2]   B:[D2]
        assert x.shape[1] == self.weight.shape[0]
        out = np.matmul(x, self.weight.getdata()) + self.bias.getdata()
        self.middle = x
        return out

    def backward(self, grad_in):
        # grad_in:[N, D2]
        assert grad_in.shape[0] == self.weight.shape[1]
        x = self.middle
        self.bias.setgrad(np.sum(grad_in, axis=0))
        self.weight.setgrad(np.matmul(x.T, grad_in) + self.reg * self.weight.getdata())
        # x.grad
        return np.matmul(grad_in, self.weight.getdata().T)

    def __repr__(self):
        return 'FullyConnect({}{})'.format(self.input, self.output)

