"""
Layer:
FullyConnect
Convolution
"""
import numpy as np
import Initialization
import Tool


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

        self.initialize()

    def __call__(self, x):
        return self.forward(x)

    def setmode(self, mode):
        self.mode = mode

    def setreg(self, reg):
        self.reg = reg

    def initialize(self):
        # judge the mode
        with_grad = False
        if self.mode == 'train':
            with_grad = True

        # init parameters
        if self.weight_init is None:
            self.weight_init = Initialization.GaussionInitialize(mean=0, std=1e-2)
        if self.bias_init is None:
            self.bias_init = Initialization.ConstantInitialize(constant=0.1)

        self.weight = Initialization.Initialize(shape=(self.input, self.output),
                                                withgrad=with_grad,
                                                initializer=self.weight_init)

        self.bias = Initialization.Initialize(shape=self.output,
                                              withgrad=with_grad,
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
        assert grad_in.shape[1] == self.weight.shape[1]
        x = self.middle
        self.bias.setgrad(np.sum(grad_in, axis=0))
        self.weight.setgrad(np.matmul(x.T, grad_in) + self.reg * self.weight.getdata())
        # x.grad
        return np.matmul(grad_in, self.weight.getdata().T)

    def __repr__(self):
        return 'FullyConnect({}{})'.format(self.input, self.output)


class Convolution:
    def __init__(self, name, in_channel, out_channel, kernel_size=3,
                 mode='train', reg=1e-4, weight_init=None, bias_init=None):
        self.name = name
        self.shape = None
        self.ksize = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode = mode
        self.reg = reg
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.middle = None
        self.param = dict()
        self.param['weight'] = dict()
        # self.param['bias'] = dict()

        self.initialize()

    def __call__(self, features):
        return self.forward(features)

    def initialize(self):
        with_grad = False
        if self.mode == 'train':
            with_grad = True
        if self.weight_init is not None:
            self.weight_init = Initialization.GaussionInitialize()
        if self.bias_init is not None:
            self.bias_init = Initialization.ConstantInitialize()

        # init the parameters
        for i in range(self.out_channel):
            self.param['weight'][i] = Initialization.Initialize((self.in_channel, self.ksize, self.ksize),
                                                                with_grad, self.weight_init)
            # self.param['bias'][i] = Initialization.Initialize(self.shape, with_grad, self.bias_init)

    def forward(self, features):
        features_out = Tool.padding(features, self.ksize)       # do padding
        h, w = features.shape[-2:]
        out = np.zeros(shape=(self.out_channel, h, w))
        # Convolution
        for k in range(self.out_channel):
            for i in range(int((self.ksize-1)/2), h+(self.ksize-1)/2):
                for j in range(int((self.ksize-1)/2), w+(self.ksize-1)/2):
                    middle = features_out[:, i-(self.ksize-1)/2:i+(self.ksize-1)/2, j-(self.ksize-1)/2:j+(self.ksize-1)/2]
                    out[k, i-(self.ksize-1)/2, j-(self.ksize-1)/2] = np.sum(middle * self.param['weight'][k].getdata())
        self.middle = features
        return out

    # def backward(self, grad_in):
    #     features = self.middle






