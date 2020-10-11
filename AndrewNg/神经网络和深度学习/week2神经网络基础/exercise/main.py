import numpy as np
import matplotlib.pyplot as plt
import h5py
import logistic_regression_as_nn as lr_nn


def load_dataset():
    """
    获取训练集和测试集

    :return: 训练集X, 训练集y, 测试集X, 训练集y, 类别
    """
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def start():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    index = 25
    plt.imshow(train_set_x_orig[index])

    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x / 255
    test_set_x = test_set_x / 255

    model = lr_nn.LogisticRegression(train_set_x, train_set_y)
    model.train()
    print(f'accuracy of training data = {model.get_accuracy()}')
    print(f'accuracy of test data = {model.accuracy(test_set_x, test_set_y)}')

    plt.show()


if __name__ == '__main__':
    start()
