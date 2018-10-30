"""
Build a Network
"""


class Network:
    def __init__(self):
        self.param = dict()
        self.layers = []
        self.loss_layer = None
        self.param_layer = []

    def __call__(self, x, label):
        return self.forward(x, label)

    def parameters(self):
        return self.param

    def train(self):
        for layer in self.layers:
            layer.mode = 'train'

    def eval(self):
        for layer in self.layers:
            layer.mode = 'test'

    def add(self, layer):
        assert layer is not None
        self.layers.append(layer)
        try:
            if layer.param is not None:
                self.param_layer.append(layer)
        except AttributeError as error:
            pass

    def add_loss(self, layer):
        self.loss_layer = layer

    def forward(self, x, label=None):
        for layer in self.layers:
            x = layer(x)
        if label is not None and self.loss_layer is not None:
            loss = self.loss_layer(x, label)
            return x, loss
        return x

    def backward(self):
        grad = self.loss_layer.backward()
        for i in range(len(self.layers)-1, 0, -1):
            grad = self.layers[i].backward(grad)
        return grad

    def optimize(self, optimizer):
        _ = self.backward()
        for layer in self.param_layer:
            for k in layer.param.keys():
                layer.param[k].data -= optimizer.optimize(layer.param[k].getgrad(), layer.name+k)

    def __repr__(self):
        string = '-----------Network----------\n'
        for i in range(len(self.layers)):
            string += "[{}]: {}".format(i, self.layers[i])
        return string

