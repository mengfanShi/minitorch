"""
Some useful tools:
Padding
ont-hot
"""
import numpy as np


def padding(feature, ksize=3):
    assert ksize % 2 == 1       # the size of kernel need to be odd

    if len(feature.shape) == 3:     # 3 dim
        h, w = feature.shape[-2:]
        out = np.zeros(shape=(feature.shape[0], h+ksize-1, w+ksize-1))
        for i in range(feature.shape[0]):
            middle = out[i, :, :]
            middle[(ksize-1)/2:h+(ksize-1)/2, (ksize-1)/2:w+(ksize-1)/2] = feature[i, :, :]
            out[i, :, :] = middle
    else:   # 2 dim
        h, w = feature.shape[-2:]
        out = np.zeros(shape=(h+ksize-1, w+ksize-1))
        out[(ksize-1)/2:h+(ksize-1)/2, (ksize-1)/2:w+(ksize-1)/2] = feature
    return out


def one_hot(label, num):
    # num means the number of kinds
    # label.shape[0] means the number of samples
    out = np.zeros((label.shape[0], num))
    out[range(label.shape[0]), label] = 1
    return out


