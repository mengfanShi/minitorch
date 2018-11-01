"""
Test the accurate of the gradient
"""
import torch
import torch.nn as nn
import numpy as np

import LossFunction
import Layer
import ActiveFunction

from  sklearn.metrics import log_loss


def test_L2():
    print('Test the accurate of L2 distance\n')
    pred = np.random.rand(6*6).reshape((6, 6)).astype('float32')
    label = np.random.rand(6*6).reshape((6, 6)).astype('float32')
    lossfunc = LossFunction.L2()
    lossfunc_torch = nn.MSELoss()
    loss = lossfunc(pred, label)

    pred_torch = torch.Tensor(pred)
    label_torch = torch.Tensor(label)
    pred_torch.requires_grad = True
    loss_torch = lossfunc_torch(pred_torch, label_torch)
    print('torch:{}, L2:{}, delta:{} '.format(loss_torch.item(), loss, (loss-loss_torch.item())))

    loss_torch.backward()
    grad_torch = pred_torch.grad.data.numpy()
    grad = lossfunc.backward()
    print(np.sum(grad-grad_torch, 0))


def test_cross_entropy_loss():
    print("gradient check: Cross Entropy")

    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    softmax = ActiveFunction.Softmax()
    x_soft = softmax(x)
    y = np.array([1, 4, 6, 3, 2], dtype='int32')
    y_onehot = np.zeros((5, 8)).astype('float32')
    y_onehot[range(0, 5), y] = 1.
    print('log loss: ', log_loss(y, x_soft, labels=[0, 1, 2, 3, 4, 5, 6, 7]))
    cross_entropy_f = LossFunction.CrossEntropy()
    cross_entropy_torch = nn.CrossEntropyLoss()

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    ce_loss_torch = cross_entropy_torch(torch_x, torch.LongTensor(y))
    ce_loss = cross_entropy_f(x_soft, y_onehot)
    print("Value:\ntorch:{},mine:{}, delta:{}"
          .format(ce_loss_torch.item(), ce_loss, (ce_loss-ce_loss_torch.item())))
    ce_loss_torch.backward()
    torch_x_grad = torch_x.grad.data.numpy()
    x_grad = softmax.backward(cross_entropy_f.backward())
    # print(np.sum(x_grad - torch_x_grad, 0))
    print(x_grad - torch_x_grad)


def test_fully_connected():
    print("gradient check: FullyConnected")

    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.array([1, 4, 6, 3, 2], dtype='int32')
    y_onehot = np.zeros((5, 12)).astype('float32')
    y_onehot[range(0, 5), y] = 1.

    # --- mine --
    fc1 = Layer.FullyConnect('fc1', 8, 10)
    fc2 = Layer.FullyConnect('fc2', 10, 12)
    relu1 = ActiveFunction.ReLU()
    softmax = ActiveFunction.Softmax()
    ce_func = LossFunction.CrossEntropy()
    fc_out1 = fc1(x)
    fc_out1 = relu1(fc_out1)
    fc_out2 = fc2(fc_out1)
    fc_out2 = softmax(fc_out2)
    sqaure_loss = ce_func(fc_out2, y_onehot)

    # --- torch ---
    weights1 = fc1.weight.getdata()
    bias1 = fc1.bias.getdata()
    weights2 = fc2.weight.getdata()
    bias2 = fc2.bias.getdata()

    torch_fc = nn.Linear(8, 10)
    torch_fc2 = nn.Linear(10, 12)
    torch_fc.weight.data.copy_(torch.Tensor(weights1.T))
    torch_fc.bias.data.copy_(torch.Tensor(bias1))
    torch_fc2.weight.data.copy_(torch.Tensor(weights2.T))
    torch_fc2.bias.data.copy_(torch.Tensor(bias2))
    torch_relu = nn.ReLU()

    torch_square_func = nn.CrossEntropyLoss()
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_fc_out = torch_fc(torch_x)
    torch_fc_out1 = torch_relu(torch_fc_out)
    torch_fc_out2 = torch_fc2(torch_fc_out1)
    torch_sqaure_loss = torch_square_func(torch_fc_out2, torch.LongTensor(y))

    print("Value:\ntorch:{}, mini:{}, delta:{}".format(
        torch_sqaure_loss.item(), sqaure_loss, (torch_sqaure_loss.item()-sqaure_loss)
    ))

    # --- my grad ---
    grad_x = ce_func.backward()
    grad_x = softmax.backward(grad_x)
    grad_fc2 = fc2.backward(grad_x)
    grad_w2 = fc2.weight.getgrad()
    grad_b2 = fc2.bias.getgrad()

    grad_x = relu1.backward(grad_fc2)
    grad_x = fc1.backward(grad_x)
    grad_w1 = fc1.weight.getgrad()
    grad_b1 = fc1.bias.getgrad()

    # --- torch grad ---
    torch_sqaure_loss.backward()
    torch_grad_x = torch_x.grad.data.numpy()
    torch_grad_w1 = torch_fc.weight.grad.data.numpy()
    torch_grad_b1 = torch_fc.bias.grad.data.numpy()
    torch_grad_w2 = torch_fc2.weight.grad.data.numpy()
    torch_grad_b2 = torch_fc2.bias.grad.data.numpy()
    print("--grad x ---")
    print(grad_x-torch_grad_x)

    print("--grad w1 ---")
    print(grad_w1-torch_grad_w1.T)

    print("--grad b1 ---")
    print(grad_b1-torch_grad_b1)

    print("--grad w2 ---")
    print(grad_w2-torch_grad_w2.T)

    print("--grad b2 ---")
    print(grad_b2-torch_grad_b2)


def test_softmax():
    print("gradient check: Softmax")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    softmax = ActiveFunction.Softmax()
    sqaure_loss_func = LossFunction.L2()

    softmax_x = softmax(x)
    square_loss = sqaure_loss_func(softmax_x, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    softmax_torch = nn.Softmax()
    square_loss_func_torch = nn.MSELoss()
    softmax_x_torch = softmax_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(softmax_x_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_softmax = sqaure_loss_func.backward()
    grad_x = softmax.backward(grad_softmax)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_sigmoid():
    print("gradient check: Sigmoid")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    sigmoid = ActiveFunction.Sigmoid()
    sqaure_loss_func = LossFunction.L2()

    sigmoid_x = sigmoid(x)
    square_loss = sqaure_loss_func(sigmoid_x, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    sigmoid_torch = nn.Sigmoid()
    square_loss_func_torch = nn.MSELoss()
    sigmoid_x_torch = sigmoid_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(sigmoid_x_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = sigmoid.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_relu():
    print("gradient check: ReLU")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    act = ActiveFunction.ReLU()
    sqaure_loss_func = LossFunction.L2()

    y_ = act(x)
    square_loss = sqaure_loss_func(y_, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    act_torch = nn.ReLU()
    square_loss_func_torch = nn.MSELoss()
    y_torch = act_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(y_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = act.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)


def test_leakyrelu():
    print("gradient check: Leaky ReLU")
    x = np.random.rand(5*8).reshape((5, 8)).astype('float32')
    y = np.random.rand(5*8).reshape((5, 8)).astype('float32')

    act = ActiveFunction.LeakyReLU(slope=0.4)
    sqaure_loss_func = LossFunction.L2()

    y_ = act(x)
    square_loss = sqaure_loss_func(y_, y)

    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    act_torch = nn.LeakyReLU(negative_slope=0.4)
    square_loss_func_torch = nn.MSELoss()
    y_torch = act_torch(torch_x)
    sqaure_loss_torch = square_loss_func_torch(y_torch, torch.Tensor(y))

    print("Value:\ntorch:{},mine:{}, delta:{}".format(sqaure_loss_torch.item(), square_loss,
                                                      (sqaure_loss_torch.item()-square_loss)))

    # --- my grad ---
    grad_sigmoid = sqaure_loss_func.backward()
    grad_x = act.backward(grad_sigmoid)

    # --- torch grad ---
    sqaure_loss_torch.backward()
    grad_x_torch = torch_x.grad.data.numpy()

    print(grad_x_torch - grad_x)

# if __name__ == '__main__':
#     test_L2()
#     test_cross_entropy_loss()
#     test_fully_connected()
#     test_leakyrelu()
#     test_relu()
#     test_softmax()
#     test_sigmoid()