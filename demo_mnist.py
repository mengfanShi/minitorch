# -*- coding:utf-8 -*-
# @TIME     :2018/11/1 17:02
# @Author   :Fan
# @File     :demo_mnist.py
import numpy as np
import Initialization
import ActiveFunction
import LossFunction
import Layer
import Optimizer
import Regularization
import Network
import Tool
from Data_Load.Mnist import trainset, testset
import tqdm     # visualize the progress

def mnist():

    weight_init = Initialization.GaussionInitialize(mean=0, std=0.1)
    bias_init = Initialization.ConstantInitialize(0.1)

    # build the network
    model = Network.Network()
    model.add(
        Layer.FullyConnect(
            name='fc1',
            feature_in=784,
            feature_out=512,
            weight_init=weight_init,
            bias_init=bias_init
        )
    )

    model.add(
        Regularization.BatchNormal(
            name='bn1',
            num_feature=512
        )
    )

    model.add(
        ActiveFunction.ReLU()
    )

    model.add(
        Layer.FullyConnect(
            name='fc2',
            feature_in=512,
            feature_out=256,
            weight_init=weight_init,
            bias_init=bias_init
        )
    )

    model.add(
        Regularization.BatchNormal(
            name='bn2',
            num_feature=256
        )
    )

    model.add(
        ActiveFunction.ReLU()
    )

    model.add(
        Layer.FullyConnect(
            name='fc3',
            feature_in=256,
            feature_out=10,
            weight_init=weight_init,
            bias_init=bias_init
        )
    )

    model.add(
        ActiveFunction.Softmax()
    )

    model.add_loss(
        LossFunction.CrossEntropy()
    )

    print(model)

    # set the optimizer
    lr = 1e-1
    optimizer = Optimizer.Momentum(lr=lr)

    # load the data
    train = trainset()
    test = testset()
    image, label = train['image'], train['label']
    test_image, test_label = test['image'], test['label']

    # set the size of batch
    batch_size = 512
    test_batch = 1000
    epochs = 100
    pbar = tqdm.tqdm(range(epochs))

    # start the train
    for epoch in pbar:
        losses = []
        model.train()
        for i in range(int(len(image)/batch_size)):
            batch_image = np.array(image[i*batch_size:(i+1)*batch_size])
            batch_label = np.array(label[i*batch_size:(i+1)*batch_size])
            batch_label = Tool.one_hot(batch_label, 10)
            _, loss = model.forward(batch_image, batch_label)
            _ = model.backward()
            model.optimize(optimizer)
            losses.append(loss)

        model.eval()
        pred = np.zeros((len(test_label)))
        for i in range(int(len(test_label)/test_batch)):
            batch_image = np.array(test_image[i*test_batch:(i+1)*test_batch])
            preds = model.forward(batch_image)
            preds = np.argmax(preds, axis=1)
            pred[i*test_batch:(i+1)*test_batch] = preds

        accurate = np.sum(test_label == pred) * 100 / len(test_label)
        pbar.set_description('epoch:{}  loss:{}  accurate:{}'.format(epoch, float(np.mean(losses)), accurate))


if __name__ == '__main__':
    mnist()


