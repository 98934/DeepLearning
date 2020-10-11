import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import deep_neural_network as d_nn


def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    if is_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y


def start():
    X_train, Y_train, X_test, Y_test = load_dataset(is_plot=True)
    # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    model = d_nn.NeuralNetwork(X_train, Y_train, param_init='he')
    model.train()
    print(model.get_accuracy())
    print(model.accuracy(X_test, Y_test))

    plt.show()


if __name__ == '__main__':
    start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
