"""
Use numpy to load the dataset of mnist
"""
import numpy as np
import struct


def read_image(filename, num):                  # num means the number of images tend to be loaded
    with open(filename, 'rb') as file:          # 'r':read  'w':write  'x':create and write  'a':write without delete
        buf = file.read()
        index = 0

        # The first sixteen bytes of the data are saved ：MagicNum, ImageNum(60000), ImageRow(28), ImageCol(28)
        _, _, _, _ = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')
        # >IIII represents loading using large order
        # I means four bytes unsigned int

        data = []
        for i in range(num):
            # Each image in mnist dataset contains 28(row)*28(col)=784 bytes
            image = np.array(struct.unpack_from('>784B', buf, index))
            index += struct.calcsize('>784B')
            image = (image/255.0).tolist()       # normalize
            data.append(image)
        return data


def read_label(filename):
    with open(filename, 'rb') as file:
        buf = file.read()
        index = 0

        # The first eight bytes of the data are saved : MagicNum, LabelNum
        _, num = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')

        label = np.array(struct.unpack_from('>%sB' % num, buf, index))
        return label


def trainset():
    image_file = './data/mnist/train-images-idx3-ubyte'
    label_file = './data/mnist/train-labels-idx1-ubyte'
    image = read_image(image_file, 60000)
    label = read_label(label_file)
    data = dict()
    data['image'] = image
    data['label'] = label
    return data


def testset():
    image_file = './data/mnist/t10k-images-idx3-ubyte'
    label_file = './data/mnist/t10k-labels-idx1-ubyte'
    image = read_image(image_file, 10000)
    label = read_label(label_file)
    data = dict()
    data['image'] = image
    data['label'] = label
    return data





